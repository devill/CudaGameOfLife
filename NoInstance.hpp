
#ifndef NO_INSTANCE_HPP
#define NO_INSTANCE_HPP

#include <string>

class NoInstance : public std::exception
{
    public:
        NoInstance(std::string argMessage) : message(argMessage) {}
        virtual ~NoInstance() throw() {}
        virtual const char* what() const throw()
        {
            return message.c_str();
        }
    private:
        std::string message;
};

class ExsistingInstance : public std::exception
{
    public:
        ExsistingInstance(std::string argMessage) : message(argMessage) {}
        virtual ~ExsistingInstance() throw() {}
        virtual const char* what() const throw()
        {
            return message.c_str();
        }
    private:
        std::string message;
};

#endif
