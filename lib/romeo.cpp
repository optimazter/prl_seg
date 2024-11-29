#include <romeo.h>



int JULIA_RUN()
{
    jl_init();

    handle_eval_string("import Pkg");
    handle_eval_string("Pkg.activate(@__DIR__)");
    try
    {
        handle_eval_string("using MriResearchTools, ROMEO");
    }
    catch(const std::exception& e)
    {
        handle_eval_string("Pkg.add([\"MriResearchTools\", \"ROMEO\"])");
        handle_eval_string("using MriResearchTools, ROMEO");
    }
    return 0;
}


jl_value_t *handle_eval_string(const char* code)
{
    jl_value_t *result = jl_eval_string(code);
    if (!result)
    {
        throw std::runtime_error("Not able to read result!");
    }
    return result;
}

int runROMEO(const char* inputPhasePath, const char* inputMaskPath, const char* outputPath)
{
    char cmdBuff[256] = {};


    sprintf(cmdBuff, "phase = readphase(\"%s\")", inputPhasePath);
    handle_eval_string(cmdBuff);

    sprintf(cmdBuff, "mask = readmag(\"%s\")", inputMaskPath);
    handle_eval_string(cmdBuff);

    handle_eval_string("unwrapped = unwrap(phase)");

    sprintf(cmdBuff, "savenii(unwrapped, \"%s\")", outputPath);
    handle_eval_string(cmdBuff);


    return 0;
}