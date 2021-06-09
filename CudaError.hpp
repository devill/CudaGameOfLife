
#ifndef CUDA_ERROR_HPP
#define CUDA_ERROR_HPP

#include <sstream>
#include <string>

#include <stdlib.h>
#include <driver_types.h>

class CudaException : public std::exception
{
    public:
        CudaException(std::string argMessage) : message(argMessage) {}
        virtual ~CudaException() throw() {}
        virtual const char* what() const throw()
        {
            return message.c_str();
        }
    private:
        std::string message;
};

static void HandleCudaError( cudaError_t err, 
                         std::string file,
                         int line ) {
    if (err != cudaSuccess) {
        std::stringstream errstream;
        errstream << "CUDA Error #" << err << " in " << file << " on line " << line << std::endl;
        throw CudaException(errstream.str());
    }    
}

#define CUDA_ERROR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))

#endif
