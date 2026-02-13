#pragma once
#include <cmath>

#ifdef __NVCC__
#define DEVICE_HOST __device__ __host__
#else
#define DEVICE_HOST
#endif

// Single macro definition for custom loss
#define DEFINE_CUSTOM_LOSS(code_block) \
namespace autodiff { namespace loss { \
    inline DEVICE_HOST auto CustomLoss::expression() { \
        Output o; \
        Target t; \
        code_block \
    } \
}}

namespace autodiff {

// Base expression template class
template <typename Derived>
struct Expr {
    DEVICE_HOST float eval(float output, float target) const {
        return static_cast<const Derived&>(*this).eval(output, target);
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        return static_cast<const Derived&>(*this).grad(output, target);
    }
};

// Terminal expressions
struct Output : public Expr<Output> {
    DEVICE_HOST float eval(float output, float target) const { return output; }
    DEVICE_HOST float grad(float output, float target) const { return 1.0f; }
};

struct Target : public Expr<Target> {
    DEVICE_HOST float eval(float output, float target) const { return target; }
    DEVICE_HOST float grad(float output, float target) const { return 0.0f; }
};

struct Constant : public Expr<Constant> {
    float value;
    DEVICE_HOST Constant(float v) : value(v) {}
    
    DEVICE_HOST float eval(float output, float target) const { return value; }
    DEVICE_HOST float grad(float output, float target) const { return 0.0f; }
};

// Binary operations
template <typename Left, typename Right>
struct Add : public Expr<Add<Left, Right>> {
    Left left;
    Right right;
    
    DEVICE_HOST Add(const Left& l, const Right& r) : left(l), right(r) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        return left.eval(output, target) + right.eval(output, target);
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        return left.grad(output, target) + right.grad(output, target);
    }
};

template <typename Left, typename Right>
struct Sub : public Expr<Sub<Left, Right>> {
    Left left;
    Right right;
    
    DEVICE_HOST Sub(const Left& l, const Right& r) : left(l), right(r) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        return left.eval(output, target) - right.eval(output, target);
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        return left.grad(output, target) - right.grad(output, target);
    }
};

template <typename Left, typename Right>
struct Mul : public Expr<Mul<Left, Right>> {
    Left left;
    Right right;
    
    DEVICE_HOST Mul(const Left& l, const Right& r) : left(l), right(r) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        return left.eval(output, target) * right.eval(output, target);
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // Product rule: d(f*g)/dx = df/dx * g + f * dg/dx
        return left.grad(output, target) * right.eval(output, target) + 
               left.eval(output, target) * right.grad(output, target);
    }
};

template <typename Left, typename Right>
struct Div : public Expr<Div<Left, Right>> {
    Left left;
    Right right;
    
    DEVICE_HOST Div(const Left& l, const Right& r) : left(l), right(r) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        float r_val = right.eval(output, target);
        return left.eval(output, target) / (r_val + 1e-10f);  // Avoid division by zero
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // Quotient rule: d(f/g)/dx = (df/dx * g - f * dg/dx) / g^2
        float l_val = left.eval(output, target);
        float r_val = right.eval(output, target);
        float l_grad = left.grad(output, target);
        float r_grad = right.grad(output, target);
        
        float safe_r = r_val + 1e-10f;  // Avoid division by zero
        return (l_grad * safe_r - l_val * r_grad) / (safe_r * safe_r);
    }
};

// Unary operations
template <typename T>
struct Square : public Expr<Square<T>> {
    T expr;
    
    DEVICE_HOST Square(const T& e) : expr(e) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        float val = expr.eval(output, target);
        return val * val;
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // d(x^2)/dx = 2x * dx/dx
        float val = expr.eval(output, target);
        return 2.0f * val * expr.grad(output, target);
    }
};

template <typename T>
struct Exp : public Expr<Exp<T>> {
    T expr;
    
    DEVICE_HOST Exp(const T& e) : expr(e) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        return expf(expr.eval(output, target));
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // d(e^x)/dx = e^x * dx/dx
        return expf(expr.eval(output, target)) * expr.grad(output, target);
    }
};

template <typename T>
struct Log : public Expr<Log<T>> {
    T expr;
    
    DEVICE_HOST Log(const T& e) : expr(e) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        float val = expr.eval(output, target);
        return logf(fmaxf(val, 1e-10f));  // Avoid log of zero
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // d(ln(x))/dx = (1/x) * dx/dx
        float val = expr.eval(output, target);
        return expr.grad(output, target) / (fmaxf(val, 1e-10f));
    }
};

template <typename T>
struct Abs : public Expr<Abs<T>> {
    T expr;
    
    DEVICE_HOST Abs(const T& e) : expr(e) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        return fabsf(expr.eval(output, target));
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // d(|x|)/dx = sign(x) * dx/dx
        float val = expr.eval(output, target);
        float sign = (val > 0) ? 1.0f : ((val < 0) ? -1.0f : 0.0f);
        return sign * expr.grad(output, target);
    }
};

template <typename T>
struct Negate : public Expr<Negate<T>> {
    T expr;
    
    DEVICE_HOST Negate(const T& e) : expr(e) {}
    
    DEVICE_HOST float eval(float output, float target) const {
        return -expr.eval(output, target);
    }
    
    DEVICE_HOST float grad(float output, float target) const {
        // d(-x)/dx = -dx/dx
        return -expr.grad(output, target);
    }
};

// Operator overloads for expression construction
template <typename T, typename U>
DEVICE_HOST auto operator+(const Expr<T>& lhs, const Expr<U>& rhs) {
    return Add<T, U>(static_cast<const T&>(lhs), static_cast<const U&>(rhs));
}

template <typename T, typename U>
DEVICE_HOST auto operator-(const Expr<T>& lhs, const Expr<U>& rhs) {
    return Sub<T, U>(static_cast<const T&>(lhs), static_cast<const U&>(rhs));
}

template <typename T, typename U>
DEVICE_HOST auto operator*(const Expr<T>& lhs, const Expr<U>& rhs) {
    return Mul<T, U>(static_cast<const T&>(lhs), static_cast<const U&>(rhs));
}

template <typename T, typename U>
DEVICE_HOST auto operator/(const Expr<T>& lhs, const Expr<U>& rhs) {
    return Div<T, U>(static_cast<const T&>(lhs), static_cast<const U&>(rhs));
}

template <typename T>
DEVICE_HOST auto operator-(const Expr<T>& expr) {
    return Negate<T>(static_cast<const T&>(expr));
}

template <typename T>
DEVICE_HOST auto square(const Expr<T>& expr) {
    return Square<T>(static_cast<const T&>(expr));
}

template <typename T>
DEVICE_HOST auto exp(const Expr<T>& expr) {
    return Exp<T>(static_cast<const T&>(expr));
}

template <typename T>
DEVICE_HOST auto log(const Expr<T>& expr) {
    return Log<T>(static_cast<const T&>(expr));
}

template <typename T>
DEVICE_HOST auto abs(const Expr<T>& expr) {
    return Abs<T>(static_cast<const T&>(expr));
}

namespace loss {


    struct MSELoss {
        DEVICE_HOST static auto expression() {
            Output o;
            Target t;
            return square(o - t);
        }
    };

    struct BCELoss {
        DEVICE_HOST static auto expression() {
            Output o;
            Target t;
            Constant one(1.0f);
            // Safe versions to avoid log(0) and log(1)
            auto safe_o = o * Constant(0.999f) + Constant(0.0005f);
            // -(t * log(o) + (1-t) * log(1-o))
            return -(t * log(safe_o) + (one - t) * log(one - safe_o));
        }
    };
    
    struct L1Loss {
        DEVICE_HOST static auto expression() {
            Output o;
            Target t;
            return abs(o - t);
        }
    };

    struct CustomLoss {
        DEVICE_HOST static auto expression();
        
    };


    struct CrossEntropyLoss {
        DEVICE_HOST static auto expression() {
            Output o;
            Target t;
            
            // Add a small epsilon to avoid log(0)
            auto safe_o = o * Constant(0.9999f) + Constant(0.00005f);
            return -t * log(safe_o);
        }
    };
    
    
    struct HuberLoss {
        DEVICE_HOST static auto expression() {
            Output o;
            Target t;
            Constant delta(1.0f);
            auto diff = o - t;
            auto abs_diff = abs(diff);
            
            // Note: This is a continuous approximation of Huber loss
            Constant half(0.5f);
            auto squared_term = half * square(diff);
            auto linear_term = delta * (abs_diff - half * delta);
            
            Constant scale(5.0f);
            auto weight = Constant(1.0f) / (Constant(1.0f) + exp(scale * (abs_diff - delta)));
            
            return weight * squared_term + (Constant(1.0f) - weight) * linear_term;
        }
    };
}

}

