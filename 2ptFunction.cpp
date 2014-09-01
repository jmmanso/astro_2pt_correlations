// This code computes the two-point correlation function of galaxies
// using OpenMP parallelization.
// 1. Opens input text file with coordinates and weights of sky pixels.
// The weights represent the number of galaxies within that pixel.
// 2. Computes all the distance pairs between pixels.
// 3. Defines a small number of angular bins, and counts the number
// of pixel pairs that fall in each one. Then, computes the 2-point correlation
// function and outputs the result to cout.


// Load some libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>                      


using namespace std;

// Define input text file  
char* input_file;

// Seed for random generator
int seed = (int)time(0);    

// Define number of angular bins as "nsteps"
#define nsteps 22

// Conversion factors between degrees and radians
double pi = 3.14159265359;
double conv = pi/180.0;
double iconv = 180.0/pi;

// Parameters that define the angular bins where the galaxy pairs will be counted
double start=-1.3;
double step=0.1;

// Allocate vectors for RA/DEC coordinates 
vector<double> ras;
vector<double> decs;

// Do the same for the mask coordinates
vector<double> ras_mask;
vector<double> decs_mask;

// Allocate pixel weight vectors
vector<double> real_weights;
vector<double> mask_weights;


// Length of the coordinate array:
int length;

// Loads the input text file, fills the coordinate vectors for the real catalog and the mask
// as well as the weights
int load_array()
{
	
	ifstream inmap;
    double ra;
    double dec;
    double weight;
	inmap.open(input_file); 
	if(! inmap.is_open() ) //check if file can be opened
	{ 
		cout << "could not open file"; return false;
	}
	while ( inmap >> ra >> dec >> weight ) //extract the values and dump them into array
	{
	  ras.push_back(ra);
      decs.push_back(dec);
      real_weights.push_back(weight);
	  mask_weights.push_back(1);
	  ras_mask.push_back(ra);
      decs_mask.push_back(dec);
	}
	inmap.close();
	cout.precision(9);
    length = real_weights.size();
	return 0;
}


//FUNCTION DISTANCES. Calculates all distances between pixels
// This uses OpenMP parallelization
vector<float> distances;
void dist()
{
       unsigned int newlength = (length  * ( length-1))/2;
       distances.resize(newlength  );
       int i; int j;
       #pragma omp parallel for default(shared) private(i,j)

         for (i=0; i<(length-1); i++)
        	{
	             for (j=(i+1); j<length; j++) 
	        	{
			   		distances.at( j+i*length-(i+1)*(2+i)/2 )= 2 * iconv * asin( sqrt (pow (sin( conv*(ras.at(i)-ras.at(j))/2 ), 2 )*cos( conv*decs.at(i) )*cos( conv*decs.at(j) ) + pow (sin (conv*(decs.at(i)-decs.at(j))/2), 2) )); 
			
			  }
		}
	 //end OMP
      
}

//FUNCTION CORRELATE. Using the distance matrix computed before, aggregate those distances in defined angular bins.
// This uses OpenMP parallelization
vector<double> corrl(vector<double>  cat1, vector<double>  cat2)
{		
    vector<double> lbinlog(nsteps);
	vector<double> lbin(nsteps);
	vector<double> steplist(nsteps);
	for (int h=0; h<nsteps; h++)
	{
		lbinlog.at(h)=start+h*step	;
		lbin.at(h)=pow ( 10 , lbinlog.at(h));
		steplist.at(h) = pow (10 ,(lbinlog.at(h)+step)) - pow (10 , lbinlog.at(h));
	}
	
	vector<double> bin(length*nsteps, 0.0);

	long int kk; long int jj; float aux_dist; long int i; long int j; long int k;
#pragma omp parallel for default(shared) private(k,i,j,kk,jj,aux_dist) 

	for (k=0; k<length; k++)
	{
		for (i=0; i<nsteps; i++)
		{
			for (j=0; j<length; j++)
			{       
                                if (k==j) continue;// k and j should NOT be allowed to be equal.
			        if (k>j) {
				          kk = j;
                                          jj = k;
                                          }//k must be smaller than j to pick the right value 
 			        if (k<j) {
                                          kk = k;
                                          jj = j;
				          }                
                               
                                aux_dist = distances.at( jj+kk*length-(kk+1)*(2+kk)/2  );

				if ((lbin.at(i) <= aux_dist) && (aux_dist <= (lbin.at(i)+steplist.at(i))))
				{
				  bin.at(k*nsteps+i)=bin.at(k*nsteps+i) + cat1.at(k) * cat2.at(j) ;
				}
				
			}
		}
	 }
	//end OMP
	vector<double> avgbin (nsteps);
	for (int m=0; m<nsteps; m++)
	{
		
		for (int r=0; r<length; r++)
		{
		    avgbin.at(m)+= bin.at(r*nsteps + m);	
		}
          
	}
	return avgbin;
}



//  Compute galaxy pair counts and output correlation function to cout. Prints a number for every angular bin

int write_tpcf()
{ 	
	vector<double> dd (nsteps);
	vector<double> mm (nsteps);
	// Compute correlation of real catalog
	dd = corrl(real_weights, real_weights);
	// Compute correlation of mask
	mm = corrl(mask_weights, mask_weights); 
	// Output results
    for (int i=0; i<nsteps; i++) {  cout << dd.at(i)<< "  ";    } cout << '\n'; 
    for (int i=0; i<nsteps; i++) {  cout << mm.at(i)<< "  ";     }cout << '\n';
	return 0;
}


//MAIN function
int main( int argc, char* argv[]  )
{
  if (argc != 2){ cout << "Incorrect number of arguments provided" << endl; exit(0);}
  input_file = argv[1];
  cout.precision(7);
  cout<<fixed;
  // Load input
  load_array();
  // Calculate distances between pixel pairs 
  dist();
  // Compute pair counts (correlation function) and output result
  write_tpcf();
  // Print how long it took
  int end = (int)time(0);
  cout << "Elapsed time: "<< (end-seed)<< " seconds" << endl;
  exit (0);
}
