#pragma once

#include <julia.h>
#include <stdexcept>

//JULIA_DEFINE_FAST_TLS 


jl_value_t *handle_eval_string(const char* code);

int JULIA_RUN();

int runROMEO(const char* inputPhasePath, const char* inputMaskPath, const char* outputPath);
