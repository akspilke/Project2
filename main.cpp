#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include "time.h"
#include <armadillo>
using namespace arma;
using namespace std;

// Finding the maximum non-diagonal element, where it is most efficient to start Jacobi's method
double findMaxOffdiagonalElement(mat &A, int &k, int &l)
{
    double offdiagMax = 0;
    for(int i=0; i<A.n_rows; i++) {
        for(int j=i+1; j<A.n_cols; j++) {
            if (fabs(A(i,j)) > offdiagMax) {
                offdiagMax=fabs(A(i,j));
                k = i;          // The row where the maximum value resides = k
                l = j;          // The column where the maximum value resides = l
            }
        }
    }
    return offdiagMax;
}

// Rotating matrix A and R around the maximum component at a_kl
void rotate(mat &A, mat &R, int k, int l) {
    //solve for tau
    double tau = (A(l,l) - A(k,k))/(2*A(k,l));

    //solve for t (tan(theta))
    double t;
    if ( tau >= 0 ) {
        t = 1.0/(tau + sqrt(1.0 + tau*tau));
    } else {
        t = -1.0/(-tau +sqrt(1.0 + tau*tau));
    }
    double c = 1/sqrt(1+t*t);
    double s = c*t;

    //perform rotation
    double a_kk = A(k,k);   // Keeping maximum elements for later, A(k,k) will be changed.
    double a_ll = A(l,l);
    A(k,k) = c*c*a_kk - 2.0*c*s*A(k,l) + s*s*a_ll;
    A(l,l) = s*s*a_kk + 2.0*c*s*A(k,l) + c*c*a_ll;

    for (int i=0; i<A.n_rows; i++){
        if (i != k && i !=l ) {

            double a_ik = A(i,k);
            double a_il = A(i,l);

            A(i,k)=c*a_ik-s*a_il;
            A(k,i) = A(i,k);

            A(i,l)=c*a_il+s*a_ik;
            A(l,i) = A(i,l);
        }
        double r_ik = R(i,k);
        double r_il = R(i,l);
        R(i,k) = c*r_ik - s*r_il;
        R(i,l) = c*r_il + s*r_ik;
    }
    A(k,l)=0; //nondiagonal element is zero
    A(l,k)=0; //nondiagonal element is zero
}
int main(int argc, char *argv[])
{
    // Setting where we want to save the output of this code
    ofstream outputFile;
    outputFile.open("non0w05.txt");
    outputFile << setiosflags(ios::showpoint | ios::uppercase);

    //question 2b- Rotation
    //2c unit tests
    // Doing the above rotations for our specialised matrix A
    int n=10;   // Defining number of grid points

    // Allocating space for vectors we will use
    double *rho = new double[n];
    double rho_max=6.0;
    double *e = new double[n];
    double *V = new double[n];
    double *d = new double[n];
    double h;
    double w = 0.5;

    mat A = zeros<mat>(n,n);     // matrix definition for Ax=lambdaX
    mat R = eye<mat>(n,n);       // Defining unit matrix where eigenvalues will be filled in along diagonal

    // Defining diagonal and off diagonal elements of our specialised matrix A
    for (int i=0; i<n; i++){
        h=rho_max/ (n+1);       // Step length
        rho[i]=(i+1)*h;         // discretized values of position rho
        V[i]= rho[i]*rho[i];                // Potential for one electron
        //V[i]= w*w*rho[i]*rho[i];            // Potential for Non-Interacting case
        //V[i]= w*w*rho[i]*rho[i] + 1/rho[i]; // Potential for Interacting case
        d[i]=2/(h*h) + V[i];    // Diagonal elements for two interacting electrons
        e[i]=-1/(h*h);          // non diagonal matrix elements

        // Filling our diagonal and non diagonal matrix elements into matrix A
        A(i,i) = d[i];
        if(i<n-1) {
            A(i+1,i) = e[i];
            A(i,i+1) = e[i];
        }
    }
    // terminates the program if A is not a square matrix
    if(A.n_rows != A.n_cols) {
        cout << "Error, not square matrix" << endl;
        terminate();
    }
    // Insterting tests into our program such that it will keep performing rotations until we are satisfied with the result
    double tolerance = 1e-8;    // Our tolerated value for zero
    int maxIteration = n*n*n;   // Maximum number of interations
    int iterations = 0;
    int k,l;
    double maxOffDiagonalElement = findMaxOffdiagonalElement(A, k, l);
    while(maxOffDiagonalElement > tolerance && iterations < maxIteration) {
        rotate(A, R, k, l);
        maxOffDiagonalElement = findMaxOffdiagonalElement(A, k, l);
        iterations++;
    }
    uvec sortedIndices = sort_index(A.diag(0));

    cout << "Finished jacobi using " << iterations << " iterations." << endl;
    cout << "The eigenvalues are " << endl;
    cout << sort(A.diag(0)) << endl;
    cout << "The eigenvector for the ground state is " << endl;

    for (int j=0; j<n; j++){
        int groundStateIndex = sortedIndices(0);
        int firstExcited = sortedIndices(1);
        int secondExcited = sortedIndices(2);
        //outputFile << setprecision(10) << setw(20) << R(groundStateIndex,j) << endl;
        double v = R(j, groundStateIndex);
        cout << v*v << ", "; //returns zero so it is orthogonal
        //outputFile << setprecision(10) << setw(20) << v*v << endl;
        }

    // Clearing memory space
    delete [] V;
    delete [] e;
    delete [] d;
    delete [] rho;

    return 0;
}

