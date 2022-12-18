#include <iostream>
#include <functional>
#include <omp.h>

#define PI 3.14159265359
#define NB_THREADS 8
using namespace std;

/*
\int_0^1 f(x) dx = \pi
Example taken from https://www.openmp.org/wp-content/uploads/SC19-Mattson-Common-Core.pdf
*/
double f(double x){
    return 4.0 / (1+x*x);
}

double fb(double x){
    return f(x)*f(x)*f(x)*f(x)*f(x) / (f(x)*f(x)*f(x)*f(x));
}

double fbb(double x){
    // "bloat" function: returns the same as f but artificially much longer to compute
    // Only when this function is used can we start to get better results from multithreaded code
    return fb(x)*fb(x)*fb(x)*fb(x)*fb(x)*fb(x)*fb(x) / (fb(x)*fb(x)*fb(x)*fb(x)*fb(x)*fb(x));
}


/*
Compute PI using a left Riemann sum -> loop over potentially HUGE nb_steps
But naive implementation introduces "loop carried dependencies"
*/ 
double calc_pi(int nb_steps){
    double step = 1.0 / nb_steps;
    double x = 0.0;  // bad
    double sum = 0;  // bad as well
    for (int i=0; i < nb_steps; i++){
        sum += fbb(x);
        x += step;
    }
    double pi = sum * step;
    return pi;
}

/*
Here, the for loop jumps by numthreads stride which migh not be cache friendly
*/
double calc_pi_omp1(int nb_steps){
    double sum[NB_THREADS];
    std::fill_n(sum, NB_THREADS, 0);

    double step = 1.0 / nb_steps;
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int numthreads = omp_get_num_threads();  // number of threads currently in the thread pool
        for (int i=id; i < nb_steps; i += numthreads){
            sum[id] += fbb(i*step);
        }
    }

    double full_sum = 0.0;
    for(int i=0; i < NB_THREADS; i++) {
        full_sum += sum[i];
    }
    
    return full_sum*step;
}


/*
Try to switch a bit the order of iteration in the arrays so that each thread iterates contiguously on the sub-arrays: 
does not change much
*/
double calc_pi_omp2(int nb_steps){
    double sum[NB_THREADS];
    std::fill_n(sum, NB_THREADS, 0);

    double step = 1.0 / nb_steps;
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int numthreads = omp_get_num_threads();  // number of threads currently in the thread pool
        int idx_start = id * nb_steps/numthreads;
        int idx_end = (id+1) * nb_steps/numthreads;
        for (int i=idx_start; i < idx_end; i++){
            sum[id] += fbb(i*step);
        }
    }

    double full_sum = 0.0;
    for(int i=0; i < NB_THREADS; i++) {
        full_sum += sum[i];
    }
    
    return full_sum*step;
}

/*
Preceding implementation are clumsy for 2 reasons:
- in the for loop, we have to take care about dividing the array ourself, computing the right indexes etc.
which is error prone and maybe not optimal in array division
- we loop over the split sum[] array ourselves once we left the parallel region

Solutions:
- For loops, the better way is to use the openmp "for" command -> openmp does internally division etc.
- private variable lsum -> each thread has its own private copy of the variable
- each time a thread has finished computation on its sub-array, the critical region enables to add its result withou
any other thread trying to write at the same time on full_sum

A tad faster than 1 and 2
*/

double calc_pi_omp3(int nb_steps){
    double step = 1.0 / nb_steps;
    double full_sum = 0.0;  // local variable with automatic storage duration declared BEFORE parallel region -> shared between threads
    double lsum;
    #pragma omp parallel private (lsum)
    {
        // lsum explicitelly declared as private for each thread
        // if not, all the threads write to the same lsum and so we end up with a quite random results
        // ACTUALLY: an alternative is to directly eclare the variable HERE :
        // double lsum
        // 
        lsum = 0.0;  
        #pragma omp for
        for (int i=0; i < nb_steps; i++){
            lsum += fbb(i*step);
        }

        #pragma omp critical
        {
            full_sum += lsum;
        }
    }
    
    return full_sum*step;
}

/*
Our problem involves a reduction operation: there is a compilator directive for that!
*/
double calc_pi_omp4(int nb_steps){
    double step = 1.0/nb_steps;
    double full_sum = 0.0;
    #pragma omp parallel for reduction (+:full_sum)
    for (int i=0; i< nb_steps; i++){
        full_sum += fbb(i*step);
    }
    return step * full_sum;
}


////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

void compute_and_time(std::function<double(double)> calc, int nb_steps, double &res, double &dt) {
    double start_time = omp_get_wtime();
    res = calc(nb_steps);
    dt = omp_get_wtime() - start_time;
}

int main () {
    cout << "Number of available threads:  " << omp_get_num_procs() << endl;
    #pragma omp parallel
    {
        #pragma omp critical  // in this section, only one thread at a time can execute (handle mutexes etc.)
        {
            cout << "Thread speaking: " << omp_get_thread_num() << endl;
        }
    }
    cout << "Parallel block finished" << endl;

    omp_set_num_threads(NB_THREADS);  // can be any number (even 9000!), default is the actual nb of CPU cores

    const int NB_STEP = 1000000;
    cout << "\nPI: " << PI << endl;
    cout << "NB_STEP: " << NB_STEP << endl;

    double res, dt;
    compute_and_time(calc_pi, NB_STEP, res, dt);
    cout << "calc_pi:       " << res << ", " << dt << endl;
    compute_and_time(calc_pi_omp1, NB_STEP, res, dt);
    cout << "calc_pi_omp1:  " << res << ", " << dt << endl;
    compute_and_time(calc_pi_omp2, NB_STEP, res, dt);
    cout << "calc_pi_omp2:  " << res << ", " << dt << endl;
    compute_and_time(calc_pi_omp3, NB_STEP, res, dt);
    cout << "calc_pi_omp3:  " << res << ", " << dt << endl;
    compute_and_time(calc_pi_omp4, NB_STEP, res, dt);
    cout << "calc_pi_omp4:  " << res << ", " << dt << endl;
}