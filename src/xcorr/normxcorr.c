/**
 * \todo generics
 *       internally using floats limits precision
 *       The input types are already converted as ndcopy.
 *       The output type and plan type should be paramterized.
 * \todo cuda
 * \todo save space if just xcorr and not normxcorr
 */
#include <math.h> // for sqrt
#include "config.h"
#include "nd.h"
#include "src/private/kind.c"

#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#ifndef restrict
#define restrict __restrict
#endif

#ifdef _MSC_VER
#define roundf(e) floorf(e+0.5f)
#endif

#define TOL (1.0e-3f)

#if 0
#define DUMP(e)  ndioClose(ndioWrite(ndioOpen(#e".h5",NULL,"w"),e))
#define DUMPR(e) ndioClose(ndioWrite(ndioOpen(#e".h5",NULL,"w"),plan->e))
#define DUMPRAS(e,name) ndioClose(ndioWrite(ndioOpen(#name".h5",NULL,"w"),plan->e))
#define DUMPR2(e) \
  { ndshape(plan->e)[1]=513; \
    ndshape(plan->e)[2]=513; \
    ndioClose(ndioWrite(ndioOpen(#e".h5",NULL,"w"),plan->e)); \
    ndshape(plan->e)[1]=512; \
    ndshape(plan->e)[2]=512; \
  }
#endif

static void breakme() {}

#define LOG(...)          printf(__VA_ARGS__)
#define REPORT(msg1,msg2) LOG("%s(%d) - %s()\n\t%s\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,msg1,msg2)
#define TRY(e)            do{if(!(e)) {REPORT(#e,"Expression evaulated as false."); breakme(); goto Error;}}while(0)
#define TRYMSG(e,msg)     do{if(!(e)) {REPORT(#e,msg); breakme(); goto Error;}}while(0)
#define NEW(T,e,N)        TRY((e)=(T*)malloc(sizeof(T)*(N)))
#define ALLOCA(T,e,N)     TRY((e)=(T*)alloca(sizeof(T)*(N)))
#define ZERO(T,e,N)       memset((e),0,sizeof(T)*(N))

#define TODO do{REPORT("TODO","Not implemented."); goto Error;} while(0)

#define countof(e) (sizeof(e)/sizeof(*(e)))
typedef size_t  stride_t;
typedef uint8_t u8;
#include "src/private/element-wise.h"



#undef LOG
#define LOG(...)          ndLogError(z,__VA_ARGS__)
// --- PLAN --------------------------------------------------------------------

/**
 * Plan
 * Each array is [2,shape(a)+shape(b)].
 */
#define NARRAY (4)
struct _ndxcorr_plan_t
{ nd_t R1,R2,R3,R4;   // four "registers" - appropriately sized temporary space
  nd_fft_plan_t plan;
  uint64_t thresh;
}; //ndxcorr_plan_t

#undef LOG
#define LOG(...) printf(__VA_ARGS__)

static unsigned nextpow2(unsigned v)
{v--;
 v |= v >> 1;
 v |= v >> 2;
 v |= v >> 4;
 v |= v >> 8;
 v |= v >> 16;
 v++;
 return v;
}

ndxcorr_plan_t ndxcorr_make_plan(nd_kind_t kind, nd_t ashape, nd_t bshape, uint64_t overlap_threshold)
{ ndxcorr_plan_t self=0;
  nd_t t=0;
  // preconditions
  TRY(ashape!=0 && bshape!=0);
  TRY(ndndim(ashape)==ndndim(bshape));
  TRY(ndtype(ashape)==ndtype(bshape));
  TRY(kind==nd_heap || kind==nd_gpu_cuda);

  NEW(struct _ndxcorr_plan_t,self,1);
  ZERO(struct _ndxcorr_plan_t,self,1);
  self->thresh=overlap_threshold;
  TRY(t=ndcast(ndref(ndunknown(ashape),0,kind),nd_f32)); // use float internally (and for output)
  // add shapes
  { size_t i=0;
    for(i=0;i<ndndim(t);++i)
      ndshape(t)[i]=nextpow2((unsigned)(ndshape(t)[i]+ndshape(bshape)[i]))+1; // must be pow2 due to fft impl, last el will be used for mirror
    ndreshape(t,ndndim(t),ndshape(t)); // reset strides
    ndShapeSet(ndInsertDim(t,0),0,2);  // complex field is dim[0] bc this is what ndfft expects
  }
  // alloc data
  { size_t d,i=0;
    nd_t *all=&self->R1;
    for(i=0;i<NARRAY;++i)
    { TRY(all[i]=ndmake(t));
      for(d=1;d<ndndim(all[i]);++d)
        ndshape(all[i])[d]--;
    }
  }
  ndfree(ndref(t,0,nd_unknown_kind));
  return self;
Error:
  return 0;
}

void ndxcorr_free_plan(ndxcorr_plan_t plan)
{ if(plan)
  { nd_t *all=&plan->R1;
    int i;
    ndfftFreePlan(plan->plan);
    for(i=0;i<NARRAY;++i)
      ndfree(all[i]);
  }
}

// --- HELPERS -----------------------------------------------------------------

/// allocates output according to plan if necessary.
static nd_t maybe_make_out(ndxcorr_plan_t plan,nd_t out)
{ nd_t shape=0;
  if(!out) 
  { TRY(shape=ndnormxcorr_output_shape(plan));
    TRY(out=ndmake_kind(shape,ndkind(plan->R1)));
  }
Error:
  ndfree(shape);
  return out;
}

/**
 * a and b are both d-dimensional arrays, possibly of different size.
 * out is a d+1 dinentional array with shape shape(a)+shape(b)
 *     The first dimension of out is the complex field.
 * a gets copied to the first plane and be gets copied to the second plane
 * The rest is filled with zeros.
 */
static nd_t form(nd_t out,nd_t a, nd_t b)
{ memset(nddata(out),0,ndnbytes(out));
  ndInsertDim(a,0);
  out=ndcopy(out,a,ndndim(a),ndshape(a));
  ndRemoveDim(a,0);
  out=ndoffset(out,0,1);
  ndInsertDim(b,0);
  out=ndcopy(out,b,ndndim(b),ndshape(b));
  ndRemoveDim(b,0);
  return ndoffset(out,0,-1);
}

/**
 * z and m have the same shape and type.
 * Does (in matlab parlance): z(m==0)=0;
 * This is only applied to array types that are compatible with ndfft, which
 * at the moment is nd_f32 and nd_f64.
 *
 * Also the arrays could be cpu or gpu based.
 */
static nd_t mask(nd_t z,nd_t m) 
{ size_t i;
  switch(ndkind(z))
  { case nd_gpu_cuda: TODO; break;
    default:
    { float *zz=(float*)nddata(z),
            *mm=(float*)nddata(m);
      TRY(ndtype(z)==nd_f32);
      for(i=0;i<ndnelem(z);++i) if(mm[i]==0) zz[i]=0.0f;
    }
  }
  return z;
Error:
  return 0;
}

/**
 * a,b,c,d are all the same shape and are complex arrays.
 * The matlab expression would look something like:
 * \code{matlab}
 *   first=@(a)   0.5.*(a+conj(reflect(a)));
 *   second=@(a) -0.5i.*(a-conj(reflect(a)));
 *   form(...
 *     first(a).*conj(second(b)),...
 *     first(c).*conj(second(d))
 * \endcode
 *
 * `reflect` is a bit complicated.  In matlab:
 * 
 * \code{matlab}
 *   function z=reflect(z)
 *     for i=1:length(size(z))
 *       z=shiftdim(flipdim(circshift(z,-1),1),1);
 *     end
 *   end
 * \endcode
 *
 * The trouble with the reflect operation is the way the zero
 * element on each dimension is treated.  To make this simpler
 * the "register" arrays are allocated with an extra element on 
 * each non-complex dimension.  The zero elements are copied to
 * the end, and the reflection iterator can then be made by using
 * negative strides and the appropriate offset.
 *
 * The only trick will be adjusting the shapes before the ffts.
 *
 * Another tricky bit is that out could overlap with one of the other arguments
 *
 */
static nd_t cross(nd_t out, nd_t a, nd_t b, nd_t c, nd_t d)
{ size_t i,n,st;
  float *aa,*bb,*cc,*dd,*oo,
        *ra,*rb,*rc,*rd;
  n=ndstrides(out)[ndndim(out)]/ndstrides(out)[1];
  st=ndstrides(out)[1]/ndstrides(out)[0]; // should be 2
  //TRY(st==2); // sanity check

  aa=((float*)nddata(a));
  ra=((float*)nddata(a))+st*(n-1);
  bb=((float*)nddata(b));
  rb=((float*)nddata(b))+st*(n-1);
  cc=((float*)nddata(c));
  rc=((float*)nddata(c))+st*(n-1);
  dd=((float*)nddata(d));
  rd=((float*)nddata(d))+st*(n-1);
  oo=((float*)nddata(out));

#define first(lhs,E,R)\
    e=(c_t*)((E)+st*i);\
    r=(c_t*)((R)-st*i);\
    (lhs).r=e->r+r->r;\
    (lhs).i=e->i-r->i
#define second(lhs,E,R)\
    e=(c_t*)((E)+st*i);\
    r=(c_t*)((R)-st*i);\
    (lhs).r=e->i+r->i;\
    (lhs).i=r->r-e->r
#define dot(lhs,a,b)\
    (lhs).r=a.r*b.r+a.i*b.i;\
    (lhs).i=a.i*b.r-a.r*b.i

  for(i=0;i<n;++i)
  { typedef struct _c_t {float r,i;} c_t;// bc complex dim is interleaved, this will get real and imag properly
    c_t
      f,s, 
      u,v,
      *e,*r;       // current element, and current reflected element - used in macros
    first(f,aa,ra);
    second(s,bb,rb);
    dot(u,f,s);
    first(f,cc,rc);
    second(s,dd,rd);
    dot(v,f,s);
    // out=u+i*v, to re-form.  The 0.25 is from a factor of 0.5 omitted from the first() and second() calculation.
    // This output step has to be last since out may overlap with a,b,c, or d.
    oo[st*i  ]=0.25f*(u.r-v.i); //real part
    oo[st*i+1]=0.25f*(u.i+v.r); //imag part
  }
  return out;
}
#undef first
#undef second
#undef dot

/** for each dim in a >0, copy first to last along dim */
static nd_t mirror(nd_t a)
{ nd_t ref=0;
  size_t i;
  for(i=1;i<ndndim(a);++i)
    ndshape(a)[i]++; // was alloc'd with one extra on each dim, so this is ok
  TRY(ref=ndunknown(a));
  ndref(ref,nddata(a),nd_static);
  for(i=1;i<ndndim(a);++i) //ignore dim 0 bc that's the complex field
  { ndPushShape(a);
    ndPushShape(ref);

    ndoffset(a,(unsigned)i,ndshape(a)[i]-1);
    ndshape(a)[i]=1;
    ndshape(ref)[i]=1;
    ndcopy(a,ref,0,0);

    ndPopShape(ref);
    ndPopShape(a);
  }
  for(i=1;i<ndndim(a);++i)
    ndshape(a)[i]--;
Finalize:
  ndfree(ref);
  return a;
Error:
  a=0;
  goto Finalize;
}

/** copies imaginary part of R to out[...,1] */
static nd_t copy_overlap_to_output(nd_t out,nd_t R)
{ int n=ndndim(out);
  TRY(ndshape(out)[n-1]==2); // assert last dim is size 2
  TRY(ndshape(R)[0]==2);   // assert first dim is size 2
  ndPushShape(out);
  ndPushShape(R);

  ndInsertDim(out,0);
  ndoffset(out,n,1);

  ndoffset(R,0,1);
  ndshape(R)[0]=1;

  ndcopy(out,R,ndndim(R),ndshape(R));
  ndPopShape(R);
  ndPopShape(out);
  return out;
Error:
  return 0;
}

// -- numerator
/* remember the overlap term is in out[...,1]
    writing to out[...,0]
    Uses R3 and R4
*/

/**
 * args array should be f1f2,f1m2,m1f2,m1m2
 * f1f1,f1m2, and m1f2 have the same strides
 * m1m2 may have it's own stride
 */
void numerator_vec_op(stride_t N, void *z,stride_t zst,
                      const size_t narg,const void **args, const stride_t *st,
                      void *param,size_t nbytes)
{ const float * restrict f1f2=(float*)args[0],
              * restrict f1m2=(float*)args[1],
              * restrict m1f2=(float*)args[2];
  float * restrict m1m2=(float*)args[3];
  float * restrict zz=(float*)z;
  const size_t a=st[0]/sizeof(float),b=st[3]/sizeof(float);
  size_t i;
  uint64_t thresh=*(uint64_t*)param;
  zst/=sizeof(float);
  for(i=0;i<N;++i)
  { float ovl=roundf(m1m2[i*b]);
    //ovl=((ovl>TOL)?ovl:TOL);
    //m1m2[i*b]=ovl;
    if(ovl>thresh)
      zz[i*zst]=f1f2[i*a]-(f1m2[i*a]*m1f2[i*a])/ovl;
    else 
      zz[i*zst]=0.0f;
  }
}

static nd_t numerator(nd_t out, ndxcorr_plan_t plan)
{ float *f1f2,*m1m2,*f1m2,*m1f2;
  const size_t n=ndstrides(plan->R3)[ndndim(plan->R3)]/ndstrides(plan->R3)[1],
              st=ndstrides(plan->R3)[1]/ndstrides(plan->R3)[0]; // should be 2
  f1f2= (float*)nddata(plan->R3);    // real part
  f1m2= (float*)nddata(plan->R4);    // real part
  m1f2=((float*)nddata(plan->R4))+1; // imag part
  ndoffset(out,ndndim(out)-1,1);
  m1m2=(float*)nddata(out);
  ndoffset(out,ndndim(out)-1,-1);
  ndInsertDim(out,0);
  { const void *args[]   ={f1f2,f1m2,m1f2,m1m2};
    const stride_t *strides[]={
      ndstrides(plan->R3),
      ndstrides(plan->R4),
      ndstrides(plan->R4),
      ndstrides(out) // m1m2
    };
    TRY(many_op(ndndim(out)-1,ndshape(out),
                nddata(out),ndstrides(out),
                4,args,strides,
                &plan->thresh,sizeof(plan->thresh),
                numerator_vec_op));
  }
  ndRemoveDim(out,0);
  return out;
Error:
  return 0;
}

/** remember the overlap term is in out[...,1]
    writing to out[...,0]
    Uses R1 and R4
*/
void denominator_vec_op(stride_t N, void *z,stride_t zst,
                      const size_t narg,const void **args, const stride_t *st,
                      void *param,size_t nbytes)
{ const float * restrict f1f1m2=(float*)args[0],
              * restrict m1f2f2=(float*)args[1],
              * restrict f1m2  =(float*)args[2],
              * restrict m1f2  =(float*)args[3],
              * restrict m1m2  =(float*)args[4];
  float * restrict zz=(float*)z;
  const size_t a=st[0]/sizeof(float),b=st[3]/sizeof(float);
  size_t i;
  zst/=sizeof(float);
  for(i=0;i<N;++i)
  { const float d1=f1f1m2[i*a]-f1m2[i*a]*f1m2[i*a]/m1m2[i*b],
                d2=m1f2f2[i*a]-m1f2[i*a]*m1f2[i*a]/m1m2[i*b],
                p =d1*d2,
                v =(p>TOL)?(float)(zz[i*zst]/sqrt(d1*d2)):0.0f;
    zz[i*zst]=(v<-1.0f)?-1.0f:((v>1.0f)?1.0f:v);
  }
}

static nd_t denominator(nd_t out, ndxcorr_plan_t plan)
{ const size_t n=ndstrides(plan->R3)[ndndim(plan->R3)]/ndstrides(plan->R3)[1],
               st=ndstrides(plan->R3)[1]/ndstrides(plan->R3)[0]; // should be 2;
  float *f1f1m2,*m1f2f2,*m1m2,*f1m2,*m1f2;
  f1f1m2= (float*)nddata(plan->R3);  // real part
  m1f2f2= (float*)nddata(plan->R3)+1;// imag part
  f1m2= (float*)nddata(plan->R4);    // real part
  m1f2=((float*)nddata(plan->R4))+1; // imag part
  ndoffset(out,ndndim(out)-1,1);
  m1m2=(float*)nddata(out);
  ndoffset(out,ndndim(out)-1,-1);
  ndInsertDim(out,0);
  { const void *args[]   ={f1f1m2,m1f2f2,f1m2,m1f2,m1m2};
    const stride_t *strides[]={
      ndstrides(plan->R3),
      ndstrides(plan->R3),
      ndstrides(plan->R4),
      ndstrides(plan->R4),
      ndstrides(out)
    };
    TRY(many_op(ndndim(out)-1,ndshape(out),
                nddata(out),ndstrides(out),
                5,args,strides,
                0,0,denominator_vec_op));
  }
  ndRemoveDim(out,0);
  return out;
Error:
  return 0;
}

nd_t square(nd_t a)
{ size_t i;
  float *aa=(float*)nddata(a);
  for(i=0;i<ndnelem(a);++i)
    aa[i]=aa[i]*aa[i];
  return a;
}

static void mask_ip_vec_op_u8(stride_t n, void *z, stride_t zst,const void* x, stride_t xst, void* p,size_t b)
{ size_t i;
  float * restrict zz=(float*)z;
  const u8* restrict xx=(u8*)x;
  zst/=sizeof(float);
  for(i=0;i<n;++i)
    if(xx[i*xst]==0)
      zz[i*zst]=0.0f;
}

static nd_t mask2(nd_t a, nd_t m1, nd_t m2)
{ TRY(ndtype(a)==nd_f32);
  TRY(ndtype(m1)==nd_u8);
  TRY(ndtype(m2)==nd_u8);
  TRY(unary_op(ndndim(m1),ndshape(m1),nddata(a),ndstrides(a)+1,nddata(m1),ndstrides(m1),0,0,mask_ip_vec_op_u8));
  ndoffset(a,0,1);
  TRY(unary_op(ndndim(m2),ndshape(m2),nddata(a),ndstrides(a)+1,nddata(m2),ndstrides(m2),0,0,mask_ip_vec_op_u8));
  ndoffset(a,0,-1);
  return a;
Error:
  return 0;
}

// --- INTERFACE ---------------------------------------------------------------

/*
 a,ma,b,mb are real with shapes `shapa` and `shapeb` used to make the plan.
 TODO
 [X] form - takes care of padding (uses post, zero-filled)
 [X] mask
 [x] mirror - copies zero element on each non-complex dimension around to the other side.
 FIXME: change shape for ffts. Do my ffts handle non-contiguous properly? I don't see why they wouldn't
 [X] cross - takes care of unpack, conj/reflect and reform
 [X] copy_overlap_to_output(out,plan->R3)
     the last dimension of out is shape2, the overlap goes in out[...,1]
 [X] numerator - numerator of the final NCC computation...
                           a careful,but straightforward iteration over shapes
                           to combine elements
 [X] denominator - do the denominator of the final NCC computation...
                           a careful,but straightforward iteration over shapes
                           to combine elements
 [X] square -- in place squaring of all elements
 */

/** Caller is respondible for freeing the returned result with ndfree.
    Does not allocate data, just returns an array with the appropriate type
    shape and kind.
 */
nd_t ndnormxcorr_output_shape(ndxcorr_plan_t plan)
{ nd_t out=ndunknown(plan->R1);
  out=ndRemoveDim(ndShapeSet(out,0,1), 0); // get rid of complex dim
  return ndShapeSet(out,ndndim(out),2);
}

nd_t ndxcorr_output_shape(ndxcorr_plan_t plan)
{ nd_t out=ndunknown(plan->R1);
  return ndRemoveDim(out,0);
}

nd_t ndxcorr(nd_t out, nd_t a, nd_t b, ndxcorr_plan_t plan)
{
  return 0;
}

nd_t ndnormxcorr(nd_t out, nd_t a, nd_t b, ndxcorr_plan_t plan)
{ // this will be exactly the same ndnormxcorr_masked, but
  // R2 will be populated by ones
  return 0;
}


nd_t ndnormxcorr_masked(nd_t out, nd_t a, nd_t ma, nd_t b, nd_t mb, ndxcorr_plan_t plan)
{ TRY(out=maybe_make_out(plan,out));
  REQUIRE(out,PTR_ARITHMETIC|CAN_MEMCPY); // once subroutines are implemented in cuda can relax this a bit
  form(plan->R1,a,b);
  form(plan->R2,ma,mb);
  mask(plan->R1,plan->R2);
  TRY(ndfft(plan->R1,plan->R1,&plan->plan));
  TRY(ndfft(plan->R2,plan->R2,&plan->plan));
  mirror(plan->R1); // tested
  mirror(plan->R2);
  cross(plan->R3,plan->R1,plan->R1, //tested
                 plan->R2,plan->R2);
  cross(plan->R4,plan->R1,plan->R2,
                 plan->R2,plan->R1);
  TRY(ndifft(plan->R3,plan->R3,&plan->plan));
  TRY(ndifft(plan->R4,plan->R4,&plan->plan)); // done with R1  
  copy_overlap_to_output(out,plan->R3); // tested
  
  numerator(out,plan); // also conditions the overlap

  form(plan->R1,a,b);  
  square(plan->R1);
  mask2(plan->R1,ma,mb);
  TRY(ndfft(plan->R1,plan->R1,&plan->plan));
  mirror(plan->R1);
  cross(plan->R3,plan->R1,plan->R2,
                 plan->R2,plan->R1);
  TRY(ndifft(plan->R3,plan->R3,&plan->plan));
  out=denominator(out,plan);
  return out; 
Error:
  return 0;
}
