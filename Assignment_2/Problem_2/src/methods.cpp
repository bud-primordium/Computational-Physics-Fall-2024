/*
@Author: Gilbert Young
@Time: 2024/09/19 08:56
@File_name: methods.cpp
@Description:
Implementation file containing definitions of the optimization methods:
1. Steepest Descent Method
2. Conjugate Gradient Method
3. Simulated Annealing
4. Genetic Algorithm
*/

#include "methods.h"
#include "functions.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <vector>

// Steepest Descent Method
Result steepestDescent(double x0, double y0, double alpha, int maxIter, double tol)
{
    Result res;
    auto start = std::chrono::high_resolution_clock::now();
    double x = x0, y = y0, dx, dy;
    res.iterations = 0;

    for (int i = 0; i < maxIter; ++i)
    {
        computeGradient(x, y, dx, dy);
        double norm = sqrt(dx * dx + dy * dy);
        if (norm < tol)
        {
            res.iterations = i;
            break;
        }
        // Update positions
        x -= alpha * dx;
        y -= alpha * dy;
        res.iterations = i + 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    res.duration = std::chrono::duration<double>(end - start).count();
    res.x = x;
    res.y = y;
    res.f = functionToMinimize(x, y);
    return res;
}

// Nonlinear Conjugate Gradient Method (Fletcher-Reeves)
Result conjugateGradient(double x0, double y0, int maxIter, double tol)
{
    Result res;
    auto start = std::chrono::high_resolution_clock::now();
    double x = x0, y = y0, dx, dy;
    computeGradient(x, y, dx, dy);
    double g0 = dx * dx + dy * dy;
    double d_x = -dx;
    double d_y = -dy;
    res.iterations = 0;

    for (int i = 0; i < maxIter; ++i)
    {
        // Line search to find optimal alpha
        double alpha = lineSearchBacktracking(x, y, d_x, d_y);
        // Update positions
        x += alpha * d_x;
        y += alpha * d_y;
        // Compute new gradient
        double new_dx, new_dy;
        computeGradient(x, y, new_dx, new_dy);
        double gk_new = new_dx * new_dx + new_dy * new_dy;
        // Check for convergence
        if (sqrt(gk_new) < tol)
        {
            res.iterations = i + 1;
            break;
        }
        // Compute beta (Fletcher-Reeves)
        double beta = gk_new / g0;
        // Update directions
        d_x = -new_dx + beta * d_x;
        d_y = -new_dy + beta * d_y;
        // Update gradient magnitude
        g0 = gk_new;
        res.iterations = i + 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    res.duration = std::chrono::duration<double>(end - start).count();
    res.x = x;
    res.y = y;
    res.f = functionToMinimize(x, y);
    return res;
}

// Simulated Annealing
Result simulatedAnnealing(double x0, double y0, double T0, double Tmin, double alpha, int maxIter)
{
    Result res;
    auto start = std::chrono::high_resolution_clock::now();
    double x = x0, y = y0, f_current = functionToMinimize(x, y);
    double T = T0;
    res.iterations = 0;

    for (int i = 0; i < maxIter && T > Tmin; ++i)
    {
        // Generate new candidate solution
        double x_new = x + ((double)rand() / RAND_MAX - 0.5);
        double y_new = y + ((double)rand() / RAND_MAX - 0.5);
        double f_new = functionToMinimize(x_new, y_new);
        double delta = f_new - f_current;

        // Accept new solution if better, or with a probability
        if (delta < 0 || exp(-delta / T) > ((double)rand() / RAND_MAX))
        {
            x = x_new;
            y = y_new;
            f_current = f_new;
        }

        // Cool down
        T *= alpha;
        res.iterations = i + 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    res.duration = std::chrono::duration<double>(end - start).count();
    res.x = x;
    res.y = y;
    res.f = f_current;
    return res;
}

// Genetic Algorithm
Result geneticAlgorithm(int populationSize, int generations, double mutationRate, double crossoverRate)
{
    Result res;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Individual> population(populationSize);
    double xMin = -10.0, xMax = 10.0;
    double yMin = -10.0, yMax = 10.0;

    // Initialize population
    for (auto &ind : population)
    {
        ind.x = xMin + (xMax - xMin) * ((double)rand() / RAND_MAX);
        ind.y = yMin + (yMax - yMin) * ((double)rand() / RAND_MAX);
        ind.fitness = functionToMinimize(ind.x, ind.y);
    }

    res.iterations = generations * populationSize;

    for (int gen = 0; gen < generations; ++gen)
    {
        // Selection (Tournament Selection)
        std::vector<Individual> newPopulation;
        for (int i = 0; i < populationSize; ++i)
        {
            int a = rand() % populationSize;
            int b = rand() % populationSize;
            Individual parent = population[a].fitness < population[b].fitness ? population[a] : population[b];
            newPopulation.push_back(parent);
        }

        // Crossover (Single-point)
        for (int i = 0; i < populationSize - 1; i += 2)
        {
            if (((double)rand() / RAND_MAX) < crossoverRate)
            {
                double alpha = (double)rand() / RAND_MAX;
                double temp_x1 = alpha * newPopulation[i].x + (1 - alpha) * newPopulation[i + 1].x;
                double temp_y1 = alpha * newPopulation[i].y + (1 - alpha) * newPopulation[i + 1].y;
                double temp_x2 = alpha * newPopulation[i + 1].x + (1 - alpha) * newPopulation[i].x;
                double temp_y2 = alpha * newPopulation[i + 1].y + (1 - alpha) * newPopulation[i].y;
                newPopulation[i].x = temp_x1;
                newPopulation[i].y = temp_y1;
                newPopulation[i + 1].x = temp_x2;
                newPopulation[i + 1].y = temp_y2;
            }
        }

        // Mutation
        for (auto &ind : newPopulation)
        {
            if (((double)rand() / RAND_MAX) < mutationRate)
            {
                ind.x += ((double)rand() / RAND_MAX - 0.5);
                ind.y += ((double)rand() / RAND_MAX - 0.5);
                // Clamp to search space
                if (ind.x < xMin)
                    ind.x = xMin;
                if (ind.x > xMax)
                    ind.x = xMax;
                if (ind.y < yMin)
                    ind.y = yMin;
                if (ind.y > yMax)
                    ind.y = yMax;
            }
            ind.fitness = functionToMinimize(ind.x, ind.y);
        }

        population = newPopulation;
    }

    // Find best individual
    Individual best = population[0];
    for (const auto &ind : population)
    {
        if (ind.fitness < best.fitness)
            best = ind;
    }

    auto end = std::chrono::high_resolution_clock::now();
    res.duration = std::chrono::duration<double>(end - start).count();
    res.x = best.x;
    res.y = best.y;
    res.f = best.fitness;
    return res;
}
