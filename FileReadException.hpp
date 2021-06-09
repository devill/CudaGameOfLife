
#ifndef FILE_READ_EXCEPTION_HPP
#define FILE_READ_EXCEPTION_HPP

#include <string>

class FileReadException : public std::exception
{
    public:
        FileReadException(std::string argMessage) : message(argMessage) {}
        virtual ~FileReadException() throw() {}
        virtual const char* what() const throw()
        {
            return message.c_str();
        }
    private:
        std::string message;
};

#endif
