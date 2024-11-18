/*
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: methods.cpp
@IDE: VSCode
@Formatter: Clang-Format
@Description: Implementation of various root-finding methods.
*/

#include "methods.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>

// Bisection Method implementation
long double bisection(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places)
{
    long double fa = f(a), fb = f(b);
    if (fa * fb >= 0)
    {
        std::cerr << "Bisection method fails. f(a) and f(b) should have opposite signs.\n";
        return NAN;
    }

    long double c = a;
    for (int i = 0; i < max_iter; ++i)
    {
        c = (a + b) / 2.0L;
        long double fc = f(c);
        std::ostringstream oss;
        oss << "Step " << i + 1 << ": [" << std::fixed << std::setprecision(decimal_places) << a << ", " << b << "]";
        iterations.push_back(oss.str());
        if ((b - a) / 2.0L < tol)
            break;
        if (fa * fc < 0)
        {
            b = c;
            fb = fc;
        }
        else
        {
            a = c;
            fa = fc;
        }
    }
    return c;
}

// Newton-Raphson Method implementation
long double newton_raphson(long double x0, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places)
{
    long double x1;
    for (int i = 0; i < max_iter; ++i)
    {
        long double fx0 = f(x0);
        long double fpx0 = f_prime(x0);
        if (fpx0 == 0.0)
        {
            std::cerr << "Newton-Raphson method fails. Derivative zero.\n";
            return NAN;
        }
        x1 = x0 - fx0 / fpx0;
        std::ostringstream oss;
        oss << "Step " << i + 1 << ": x0 = " << std::fixed << std::setprecision(decimal_places) << x0
            << ", x1 = " << x1;
        iterations.push_back(oss.str());
        if (fabs(x1 - x0) < tol)
            break;
        x0 = x1;
    }
    return x1;
}

// Hybrid Method implementation
long double hybrid_method(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places)
{
    long double fa = f(a), fb = f(b);
    if (fa * fb >= 0)
    {
        std::cerr << "Hybrid method fails. f(a) and f(b) should have opposite signs.\n";
        return NAN;
    }

    long double c = a;
    for (int i = 0; i < max_iter; ++i)
    {
        c = (a + b) / 2.0L;
        long double fc = f(c);
        std::ostringstream oss;
        oss << "Step " << i + 1 << ": [" << std::fixed << std::setprecision(decimal_places) << a << ", " << b << "]";
        iterations.push_back(oss.str());
        if ((b - a) / 2.0L < tol)
            break;

        long double fpc = f_prime(c);
        if (fpc != 0.0)
        {
            long double d = c - fc / fpc;
            if (d > a && d < b)
            {
                long double fd = f(d);
                std::ostringstream oss_newton;
                oss_newton << "Newton Step: c = " << std::fixed << std::setprecision(decimal_places) << c
                           << ", d = " << d;
                iterations.push_back(oss_newton.str());
                if (fabs(d - c) < tol)
                    return d;
                if (fa * fd < 0)
                {
                    b = d;
                    fb = fd;
                }
                else
                {
                    a = d;
                    fa = fd;
                }
                continue;
            }
        }

        // Fallback to bisection
        if (fa * fc < 0)
        {
            b = c;
            fb = fc;
        }
        else
        {
            a = c;
            fa = fc;
        }
    }
    return c;
}

// Brent's Method implementation
long double brent_method(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places)
{
    long double fa = f(a), fb = f(b);
    if (fa * fb >= 0)
    {
        std::cerr << "Brent's method fails. f(a) and f(b) should have opposite signs.\n";
        return NAN;
    }

    if (fabs(fa) < fabs(fb))
    {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    long double c = a, fc = fa, s = b, fs = fb;
    bool mflag = true;
    long double d = 0.0;

    for (int i = 0; i < max_iter; ++i)
    {
        if (fb != fc && fa != fc)
        {
            // Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                (c * fa * fb) / ((fc - fa) * (fc - fb));
        }
        else
        {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions to accept s
        bool condition1 = (s < (3 * a + b) / 4.0L || s > b);
        bool condition2 = (mflag && fabs(s - b) >= fabs(b - c) / 2.0L);
        bool condition3 = (!mflag && fabs(s - b) >= fabs(c - d) / 2.0L);
        bool condition4 = (mflag && fabs(b - c) < tol);
        bool condition5 = (!mflag && fabs(c - d) < tol);

        if (condition1 || condition2 || condition3 || condition4 || condition5)
        {
            // Bisection method
            s = (a + b) / 2.0L;
            mflag = true;
        }
        else
        {
            mflag = false;
        }

        long double fs_new = f(s);
        std::ostringstream oss;
        oss << "Step " << i + 1 << ": a = " << std::fixed << std::setprecision(decimal_places) << a
            << ", b = " << b << ", s = " << s;
        iterations.push_back(oss.str());

        d = c;
        c = b;
        fc = fb;

        if (fa * fs_new < 0)
        {
            b = s;
            fb = fs_new;
        }
        else
        {
            a = s;
            fa = fs_new;
        }

        if (fabs(fa) < fabs(fb))
        {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        if (fabs(b - a) < tol)
            break;
    }

    return b;
}

// Ridder's Method implementation
long double ridder_method(long double a, long double b, long double tol, int max_iter, std::vector<std::string> &iterations, int decimal_places)
{
    long double fa = f(a), fb = f(b);
    if (fa * fb >= 0)
    {
        std::cerr << "Ridder's method fails. f(a) and f(b) should have opposite signs.\n";
        return NAN;
    }

    for (int i = 0; i < max_iter; ++i)
    {
        long double c = 0.5L * (a + b);
        long double fc = f(c);
        long double s_sq = fc * fc - fa * fb;
        if (s_sq < 0.0)
        {
            std::cerr << "Ridder's method fails. Square root of negative number.\n";
            return NAN;
        }
        long double s = sqrt(s_sq);
        if (s == 0.0)
            return c;

        long double sign = ((fa - fb) < 0) ? -1.0L : 1.0L;
        long double x = c + (c - a) * fc / s * sign;
        long double fx = f(x);
        std::ostringstream oss;
        oss << "Step " << i + 1 << ": [" << std::fixed << std::setprecision(decimal_places) << a << ", " << b << "]";
        iterations.push_back(oss.str());

        if (fabs(fx) < tol)
            return x;

        if (fc * fx < 0.0)
        {
            a = c;
            fa = fc;
            b = x;
            fb = fx;
        }
        else if (fa * fx < 0.0)
        {
            b = x;
            fb = fx;
        }
        else
        {
            a = x;
            fa = fx;
        }

        if (fabs(b - a) < tol)
            break;
    }

    return 0.5L * (a + b);
}
