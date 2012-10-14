#include <stdio.h>
#include <gmp.h>
#include <cump/cump.cuh>


#define N  1024


void  gmp_axpy (int  n, mpf_t  a, mpf_t  X [], mpf_t  Y []);
void  cump_axpy (int  n, cumpf_array_t  a, cumpf_array_t  X, cumpf_array_t  Y);


int  main ()
{
  int  i;
  int  prec = 1024;   /* precision in bits */
  int  seed = 341;    /* random seed*/

  gmp_randstate_t  rstate;

  mpf_t  a, X [N], Y [N];
  cumpf_array_t  a_, X_, Y_;

  /* set default precision of both libraries */
  mpf_set_default_prec (prec);
  cumpf_set_default_prec (prec);

  /* initialize mpf_t variables/arrays and set random numbers into them */
  gmp_randinit_default (rstate);
  gmp_randseed_ui (rstate, seed);

  mpf_init (a);
  mpf_urandomb (a, rstate, prec);

  for (i = 0;  i < N;  ++i)
    {
      mpf_init (X [i]);
      mpf_urandomb (X [i], rstate, prec);
      mpf_init (Y [i]);
      mpf_urandomb (Y [i], rstate, prec);
    }

  gmp_randclear (rstate);

  /* initialize cumpf_t variables/arrays and set mpf_t ones into them */
  cumpf_array_init_set_mpf (a_, &a, 1);
  cumpf_array_init_set_mpf (X_, X, N);
  cumpf_array_init_set_mpf (Y_, Y, N);

  printf ("Calculation (vector length = %d):\n", N);
  gmp_printf ("\t\t  |%Ff\t|   |%Ff\t|\n", X [0], Y [0]);
  gmp_printf ("\t\t  |%Ff\t|   |%Ff\t|\n", X [1], Y [1]);
  gmp_printf ("\t\t  |    :\t|   |    :\t|\n");
  gmp_printf ("%Ff\tx |    :\t| + |    :\t|\n", a);
  gmp_printf ("\t\t  |    :\t|   |    :\t|\n");
  gmp_printf ("\t\t  |%Ff\t|   |%Ff\t|\n\n", X [N-1], Y [N-1]);

  /* run axpy */
  gmp_axpy (N, a, X, Y);
  cump_axpy (N, a_, X_, Y_);

  printf ("Result:\n");
  gmp_printf ("\t|%Ff\t|\n", Y [0]);
  gmp_printf ("\t|%Ff\t|\n", Y [1]);
  gmp_printf ("\t|    :\t\t|\n");
  gmp_printf ("\t|    :\t\t|\n");
  gmp_printf ("\t|    :\t\t|\n");
  gmp_printf ("\t|%Ff\t|\n\n", Y [N-1]);

  /* compare both results */
  printf ("Comparison of both results: ");
  mpf_array_set_cumpf (X, Y_, N);
  for (i = 0;  i < N;  ++i)
    {
      if (mpf_cmp (X [i], Y [i]) != 0)
        {
          printf ("failed!\n\n");
          goto  _Exit;
        }
    }
  printf ("matched!\n\n");

_Exit:
  /* finalize */
  cumpf_array_clear (a_);
  cumpf_array_clear (X_);
  cumpf_array_clear (Y_);

  mpf_clear (a);

  for (i = 0;  i < N;  ++i)
    {
      mpf_clear (X [i]);
      mpf_clear (Y [i]);
    }

  return  0;
}


void  gmp_axpy (int  n, mpf_t  a, mpf_t  X [],  mpf_t  Y [])
{
  int  i;
  mpf_t  aX;

  mpf_init (aX);

  for (i = 0;  i < n;  ++i)
    {
      mpf_mul (aX, a, X [i]);
      mpf_add (Y [i], aX, Y [i]);
    }

  mpf_clear (aX);
}


using cump::mpf_array_t;

__global__
void  cump_axpy_kernel (int  n, mpf_array_t  a, mpf_array_t  X, mpf_array_t  Y, mpf_array_t  aX)
{
  using namespace cump;

  int  idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    {
      /* Due to CUMP's limitation,
         arithmetic between cump::mpf_array_t and cump::mpf_t is unavailable */
      mpf_mul (aX [idx], a [0], X [idx]);
      mpf_add (Y [idx], aX [idx], Y [idx]);
    }
}


void  cump_axpy (int  n, cumpf_array_t  a, cumpf_array_t  X, cumpf_array_t  Y)
{
  int  threads = 32;
  int  blocks = n / threads + (n % threads ? 1 : 0);
  cumpf_array_t  aX;

  cumpf_array_init (aX, n);
  cump_axpy_kernel <<<blocks, threads>>> (n, a, X, Y, aX);
  cumpf_array_clear (aX);
}
