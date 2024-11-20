#ifndef ROMEO_H
#define ROMEO_H

#include <julia.h>


JULIA_DEFINE_FAST_TLS 



extern "C"
{



jl_value_t *handle_eval_string(const char* code) {
    jl_value_t *result = jl_eval_string(code);
    assert(result);
    return result;
}


int JULIA_RUN()
{
    jl_init();

    handle_eval_string("import Pkg");
    handle_eval_string("Pkg.activate(@__DIR__)");

    handle_eval_string("using MriResearchTools");
    return 0;
}



int c_romeo(const char* inputPhasePath, const char* inputMagPath, const char* outputPath)
{
    char cmdBuff[256] = {};


    sprintf(cmdBuff, "phase = readphase(\"%s\")", inputPhasePath);
    handle_eval_string(cmdBuff);

    sprintf(cmdBuff, "mag = readmag(\"%s\")", inputMagPath);
    handle_eval_string(cmdBuff);

    handle_eval_string("unwrapped = unwrap(phase; mag=mag)");

    sprintf(cmdBuff, "savenii(unwrapped, \"%s\")", outputPath);
    handle_eval_string(cmdBuff);


    return 0;
}

}



#endif
