/**
 * \file
 * Macros!  Commong macros for exception handling, allocation, and other fun stuff.
 */
/// @cond DEFINES
#ifdef _MSC_VER
#define PATHSEP "\\"
#else
#define PATHSEP "/"
#endif
#define countof(e) (sizeof(e)/sizeof(*e))
#define ENDL                  "\n"
#define LOG(...)              printf(__VA_ARGS__)
#define TRY(e)                do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define HTRY(e)               do{if((e)<0){ LOG("%s(%d): %s()"ENDL "\tHDF5 Expression evaluated to a negative value."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0) 
#define TRYMSG(e,msg)         do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%sENDL",__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL(msg)             do{ LOG("%s(%d): %s()"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg); goto Error;} while(0)
#define RESIZE(type,e,nelem)  TRY((e)=(type*)realloc((e),sizeof(type)*(nelem)))
#define NEW(type,e,nelem)     TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)           if(e){free(e); (e)=NULL;}
#define STACK_ALLOC(T,e,N)    TRY(((e)=(T*)alloca((N)*sizeof(T))))
/// @endcond