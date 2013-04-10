#define ROOTS  (4096)
#define PI     (3.14159265358979323846)
#include <stdlib.h>
#include <math.h>
#include "nd.h" //for type ids

void breakme() {}

template<class T> 
inline void swap(T* a, T*b) { T t=*a;*a=*b;*b=t; }

template<class T> 
static void gather(size_t n, T* x, size_t stride)
{ size_t j=0;
  for(size_t i=0;i<n;++i)
  { size_t mask=n;      // bit mask
    if(i>j) swap(x+i*stride,x+j*stride);
    while(j&(mask>>=1)) // while bit is set
      j&=~mask;         // drop bit
    j|=mask;            // set bit
  }
}

template<class T>
inline void root_(size_t i, T* real, T* imag)
{ double th=PI/(double)i,
            s=sin(th*0.5);
  *real=-2.0*s*s;
  *imag= sin(th);
}

template<class T>
static void root(size_t i, T* real, T* imag, int direction)
{ static double table[2*ROOTS]; // real,imag interleaved
  static int inited=0;
  if(!inited) // compute table
  { for(size_t j=0;j<ROOTS;++j)
      root_(j,table+2*j,table+2*j+1);
    inited=1;
  }
  if(i<ROOTS)
  { *real=(T) (table[2*i]);
    *imag=(T) (table[2*i+1]);
  } else
  { root_(i,real,imag);
  }
  *imag *= direction; // sign flip since sin is odd; real part is even
}

template<class T>
static void inline cmul(T& r, T& i, const T& ar, const T& ai, const T& br, const T& bi)
{ r=ar*br-ai*bi;
  i=ar*bi+ai*br;
}

#define REAL(v_,i_) (((v_)+(i_)*vstride)[0])
#define IMAG(v_,i_) (((v_)+(i_)*vstride+cstride)[0])

template<class T>
static int fft1d_ip_(size_t n,T* x,size_t cstride, size_t vstride, int direction)
{ 
  gather(n,x        ,vstride);    // gather real values
  gather(n,x+cstride,vstride);    // gather imag values
  for(size_t i=1;i<n;i<<=1)       // Divide the operating range by two each time
  { size_t step=i<<1;
    T freal=1.0, 
      fimag=0.0;
    for(size_t j=0;j<i;++j)       // Select the part of the range
    { for(size_t k=j;k<n;k+=step) // Iterate through selected elements
      { size_t ii=k+i;
        T real,imag;
        cmul(real,imag,freal,fimag,REAL(x,ii),IMAG(x,ii));
        REAL(x,ii) = REAL(x,k)-real;
        IMAG(x,ii) = IMAG(x,k)-imag;
        REAL(x,k )+=real;
        IMAG(x,k )+=imag;
      }
      { T real,imag,tr,ti;
        root(i,&real,&imag,direction);
        cmul(tr,ti,freal,fimag,real,imag);
        freal+=tr;
        fimag+=ti;
      }
    }
  }
  return 1;
}

extern "C"
{
  int fft1d_ip(size_t n, nd_type_id_t tid, void *d, size_t cstride, size_t vstride, int direction)
  {
    switch(tid)
    { case nd_f32: return fft1d_ip_(n, (float*)d, cstride, vstride, direction);
      case nd_f64: return fft1d_ip_(n, (double*)d, cstride, vstride, direction);
      default: return 0;
    }
  }
}