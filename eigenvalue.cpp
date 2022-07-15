
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;


const double SMALL = 1.0E-30;          
const double NEARZERO = 1.0E-10;       

using cmplx  = complex<double>;        // complex number
using vec    = vector<cmplx>;          
using matrix = vector<vec>;            


// Prototypes
matrix readMatrix();
int howBig( string filename );
void printMatrix( string title, const matrix &A );

matrix matMul( const matrix &A, const matrix &B );
matrix matSca( cmplx c, const matrix &A );
matrix matLin( cmplx a, const matrix &A, cmplx b, const matrix &B );
vec matVec( const matrix &A, const vec &V );
vec vecSca( cmplx c, const vec &V );
vec vecLin( cmplx a, const vec &U, cmplx b, const vec &V );
double vecNorm( const vec &V );
double matNorm( const matrix &A );
double subNorm( const matrix &T );
matrix identity( int N );
matrix hermitianTranspose( const matrix &A );

cmplx shift( const matrix &A );
void Hessenberg( const matrix &A, matrix &P, matrix &H );
void QRFactoriseGivens( const matrix &A, matrix &Q, matrix &R );
void QRHessenberg( const matrix &A, matrix &P, matrix &T );
bool eigenvectorUpper( const matrix &T, matrix &E );


//========


int main()
{
   // Read matrix
   matrix A = readMatrix();
   printMatrix( "\nOriginal matrix:", A );

   int N = A.size();
   matrix P, T, E;

   // Compute eigenvalues by QR-Hessenberg method. 
   
   QRHessenberg( A, P, T );
   cout << "\nEigenvalues by QR algorithm are:\n";
   for ( int L = 0; L < N; L++ ) cout << T[L][L] << '\n';

   // Compute eigenvectors
   bool OK = eigenvectorUpper( T, E );           // Find the eigenvectors of T
   E = matMul( P, E );                           // Rotate eigenvectors to those of A
   for ( int L = 0; L < N; L++ )
   {
      cmplx lambda = T[L][L];
      vec V( N );
      for ( int j = 0; j < N; j++ ) V[j] = E[j][L];

      cout << "\n\nEigenvalue " << lambda << "\nEigenvector:\n";
      for ( int j = 0; j < N; j++ ) cout << V[j] << '\n';

      // Check matrix norm of   A v - lambda v
      cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
   }
}

//========

matrix readMatrix()                    // Input the matrix
{
   string filename = "matrix.dat";

   // Determine how large the matrix is and set array sizes accordingly
   int N = howBig( filename );         
   matrix A( N, vec(N) );             

   // Read from file
   ifstream in( filename );   assert( in );
   for ( int i = 0; i < N; i++ )
   {
      for ( int j = 0; j < N; j++ ) in >> A[i][j];
   }
   in.close();

   return A;
}

//========

int howBig( string filename )          // Reads one line to count the elements in it
{
   string s;
   ifstream in( filename );   assert( in );
   getline( in, s );
   in.close();

   stringstream ss( s );               // Creates a stream from this line
   int N = 0;
   cmplx dummy;
   while( ss >> dummy ) N++;           // Increments N for as many values as present

   return N;
}

//========

void printMatrix( string title, const matrix &A )
{
   cout << title;   if ( title != "" ) cout << '\n';

   int m = A.size(), n = A[0].size();            // A is an m x n matrix

   for ( int i = 0; i < m; i++ )
   {
      for ( int j = 0; j < n; j++ )
      {
         double x = A[i][j].real();   if ( abs( x ) < NEARZERO ) x = 0.0;
         double y = A[i][j].imag();   if ( abs( y ) < NEARZERO ) y = 0.0;
         cout << cmplx( x, y ) << '\t';
      }
      cout << '\n';
   }
}

//========

matrix matMul( const matrix &A, const matrix &B )          // Matrix times matrix
{
   int mA = A.size(),   nA = A[0].size();
   int mB = B.size(),   nB = B[0].size();   assert( mB == nA );
   matrix C( mA, vec( nB, 0.0 ) );

   for ( int i = 0; i < mA; i++ )
   {
      for ( int j = 0; j < nB; j++ )
      {
         for ( int k = 0; k < nA; k++ ) C[i][j] += A[i][k] * B[k][j];
      }
   }
   return C;
}

//========

matrix matSca( cmplx c, const matrix &A )                  // Scalar multiple of matrix
{
   int m = A.size(),   n = A[0].size();
   matrix C = A;

   for ( int i = 0; i < m; i++ )
   {
      for ( int j = 0; j < n; j++ ) C[i][j] *= c;
   }
   return C;
}

//========

matrix matLin( cmplx a, const matrix &A, cmplx b, const matrix &B )  // Linear combination of matrices
{
   int m = A.size(),   n = A[0].size();   assert( B.size() == m && B[0].size() == n );
   matrix C = matSca( a, A );

   for ( int i = 0; i < m; i++ )
   {
      for ( int j = 0; j < n; j++ ) C[i][j] += b * B[i][j];
   }
   return C;
}

//========

vec matVec( const matrix &A, const vec &V )                // Matrix times vector
{
   int mA = A.size(),   nA = A[0].size();
   int mV = V.size();   assert( mV == nA );
   vec C( mA, 0.0 );

   for ( int i = 0; i < mA; i++ )
   {
      for ( int k = 0; k < nA; k++ ) C[i] += A[i][k] * V[k];
   }
   return C;
}

//========

vec vecSca( cmplx c, const vec &V )                        // Scalar multiple of vector
{
   int n = V.size();
   vec W = V;
   for ( int j = 0; j < n; j++ ) W[j] *= c;
   return W;
}

//========

vec vecLin( cmplx a, const vec &U, cmplx b, const vec &V ) // Linear combination of vectors
{
   int n = U.size();   assert( V.size() == n );
   vec W = vecSca( a, U );
   for ( int j = 0; j < n; j++ ) W[j] += b * V[j];
   return W;
}

//========
double vecNorm( const vec &V )                             // Complex vector norm
{
   int n = V.size();
   double result = 0.0;
   for ( int j = 0; j < n; j++ ) result += norm( V[j] );
   return sqrt( result );
}

//========
double matNorm( const matrix &A )                          // Complex matrix norm
{
   int m = A.size(),   n = A[0].size();
   double result = 0.0;
   for ( int i = 0; i < m; i++ )
   {
      for ( int j = 0; j < n; j++ ) result += norm( A[i][j] );
   }
   return sqrt( result );
}

//========

double subNorm( const matrix &T )                          // Below leading diagonal of square matrix
{
   int n = T.size();   assert( T[0].size() == n );
   double result = 0.0;
   for ( int i = 1; i < n; i++ )
   {
      for ( int j = 0; j < i; j++ ) result += norm( T[i][j] );
   }
   return sqrt( result );
}

//======== 
matrix identity( int N )                                   // N x N Identity matrix
{
   matrix I( N, vec( N, 0.0 ) );
   for ( int i = 0; i < N; i++ ) I[i][i] = 1.0;
   return I;
}

//========

matrix hermitianTranspose( const matrix &A )               // Hermitian transpose
{
   int m = A.size(),   n = A[0].size();
   matrix AH( n, vec( m ) );                               // Note: transpose is an n x m matrix
   for ( int i = 0; i < n; i++ )
   {
      for ( int j = 0; j < m; j++ ) AH[i][j] = conj( A[j][i] );
   }
   return AH;
}

//========

cmplx shift( const matrix &A )                             // Wilkinson shift in QR algorithm
{
   int N = A.size();
   cmplx s = 0.0;
   int i = N - 1;
// while ( i > 0 && abs( A[i][i-1] ) < NEARZERO ) i--;     // Deflation (not sure about this)

   if ( i > 0 )
   {
      cmplx a = A[i-1][i-1], b = A[i-1][i], c = A[i][i-1], d = A[i][i];        // Bottom-right elements
      cmplx delta = sqrt( ( a + d ) * ( a + d ) - 4.0 * ( a * d - b * c ) ); 
      cmplx s1 = 0.5 * ( a + d + delta );
      cmplx s2 = 0.5 * ( a + d - delta );
      s = ( norm( s1 - d ) < norm( s2 - d ) ? s1 : s2 );
   }
   return s;
}

//========

void Hessenberg( const matrix &A, matrix &P, matrix &H )

{
   int N = A.size();

   H = A;
   P = identity( N );

   for ( int k = 0; k < N - 2; k++ )             // k is the working column
   {
      // X vector, based on the elements from k+1 down in the kth column
      double xlength = 0;
      for ( int i = k + 1; i < N; i++ ) xlength += norm( H[i][k] );
      xlength = sqrt( xlength );

      // U vector ( normalise X - rho.|x|.e_k )
      vec U( N, 0.0 );
      cmplx rho = 1.0, xk = H[k+1][k];
      double axk = abs( xk );
      if ( axk > NEARZERO ) rho = -xk / axk;
      U[k+1] = xk - rho * xlength;
      double ulength = norm( U[k+1] );
      for ( int i = k + 2; i < N; i++ )
      {
         U[i] = H[i][k];
         ulength += norm( U[i] );
      }
      ulength = max( sqrt( ulength ), SMALL );
      for ( int i = k + 1; i < N; i++ ) U[i] /= ulength;

      // Householder matrix: P = I - 2 U U*T
      matrix PK = identity( N );
      for ( int i = k + 1; i < N; i++ )
      {
         for ( int j = k + 1; j < N; j++ ) PK[i][j] -= 2.0 * U[i] * conj( U[j] );
      }

      // Transform as PK*T H PK.   Note: PK is unitary, so PK*T = P
      H = matMul( PK, matMul( H, PK ) );
      P = matMul( P, PK );
   }
}


//========


void QRFactoriseGivens( const matrix &A, matrix &Q, matrix &R )
{
   // Factorises a Hessenberg matrix A as QR, where Q is unitary and R is upper triangular
   // Uses N-1 Givens rotations
   int N = A.size();

   Q = identity( N );
   R = A;

   for ( int i = 1; i < N; i++ )       // i is the row number
   {
      int j = i - 1;                   // aiming to zero the element one place below the diagonal
      if ( abs( R[i][j] ) < SMALL ) continue;

      // Form the Givens matrix        
      cmplx c =        R[j][j]  ;           
      cmplx s = -conj( R[i][j] );                       
      double length = sqrt( norm( c ) + norm( s ) );    
      c /= length;               
      s /= length;               
      cmplx cstar = conj( c );         //  G*T = ( c* -s )     G = (  c  s  )     <--- j
      cmplx sstar = conj( s );         //        ( s*  c )         ( -s* c* )     <--- i
      matrix RR = R;
      matrix QQ = Q;
      for ( int m = 0; m < N; m++ ) 
      {
         R[j][m] = cstar * RR[j][m] - s     * RR[i][m];
         R[i][m] = sstar * RR[j][m] + c     * RR[i][m];    // Should force R[i][j] = 0.0
         Q[m][j] = c     * QQ[m][j] - sstar * QQ[m][i];
         Q[m][i] = s     * QQ[m][j] + cstar * QQ[m][i];
      }
   }
}

//========

void QRHessenberg( const matrix &A, matrix &P, matrix &T )
// Apply the QR algorithm to the matrix A. 

{
   const int ITERMAX = 10000;
   const double TOLERANCE = 1.0e-10;

   int N = A.size();

   matrix Q( N, vec( N ) ), R( N, vec( N ) ), Told( N, vec( N ) );
   matrix I = identity( N );

   // Stage 1: transform to Hessenberg matrix ( T = Hessenberg matrix, P = unitary transformation )
   Hessenberg( A, P, T );


   // Stage 2: apply QR factorisation (using Givens rotations)
   int iter = 1;
   double residual = 1.0;
   while( residual > TOLERANCE && iter < ITERMAX )
   {
      Told = T;

      // Spectral shift
      cmplx mu = shift( T );
      if ( abs( mu ) < NEARZERO ) mu = 1.0;   // prevent unitary matrices causing a problem
      T = matLin( 1.0, T, -mu, I );

      // Basic QR algorithm by Givens rotation
      QRFactoriseGivens( T, Q, R );
      T = matMul( R, Q );
      P = matMul( P, Q );

      // Reverse shift
      T = matLin( 1.0, T, mu, I );

      // Calculate residuals
      residual = matNorm( matLin( 1.0, T, -1.0, Told ) );            // change on iteration
      residual += subNorm( T );                                      // below-diagonal elements
//    cout << "\nIteration: " << iter << "   Residual: " << residual << endl;
      iter++;
   }
   cout << "\nQR iterations: " << iter << "   Residual: " << residual << endl;
   if ( residual > TOLERANCE ) cout << "***** WARNING ***** QR algorithm not converged\n";
}

//========

bool eigenvectorUpper( const matrix &T, matrix &E )
// Find the eigenvectors of upper-triangular matrix T; returns them as column vectors of matrix E
// The eigenvalues are necessarily the diagonal elements of T
// NOTE: if there are repeated eigenvalues, then THERE MAY NOT BE N EIGENVECTORS
{
   bool fullset = true;
   int N = T.size();
   E = matrix( N, vec( N, 0.0 ) );               // Columns of E will hold the eigenvectors

   matrix TT = T;
   for ( int L = N - 1; L >= 0; L-- )            // find Lth eigenvector, working from the bottom
   {
      bool ok = true;
      vec V( N, 0.0 );
      cmplx lambda = T[L][L];
      for ( int k = 0; k < N; k++ ) TT[k][k] = T[k][k] - lambda;          // TT = T - lambda I
                                                                          // Solve TT.V = 0
      V[L] = 1.0;                                // free choice of this component
      for ( int i = L - 1; i >= 0; i-- )         // back-substitute for other components
      {
         V[i] = 0.0;
         for ( int j = i + 1; j <= L; j++ ) V[i] -= TT[i][j] * V[j];
         if ( abs( TT[i][i] ) < NEARZERO )       // problem with repeated eigenvalues
         {
            if ( abs( V[i] ) > NEARZERO ) ok = false;     // incomplete set; use the lower-L one only
            V[i] = 0.0;
         }
         else
         {
            V[i] = V[i] / TT[i][i];
         }
      }

      if ( ok )
      {
         // Normalise
         double length = vecNorm( V );    
         for ( int i = 0; i <= L; i++ ) E[i][L] = V[i] / length;
      }
      else
      {
         fullset = false;
         for ( int i = 0; i <= L; i++ ) E[i][L] = 0.0;
      }
   }

   if ( !fullset )
   {
      cout << "\n***** WARNING ***** Can't find N independent eigenvectors\n";
      cout << "   Some will be set to zero\n";
   }

   return fullset;
}