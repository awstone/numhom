#include "mpi.h"
#include "petscviewer.h"
#include <petscsys.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>

/* exact solution to the oscillatory problem */
static PetscErrorCode oscillatory_u
(PetscInt dim, PetscReal time, const PetscReal x[],
 PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal eps = 0.03125;
  
  u[0] = x[0] - x[0]*x[0]
    + (eps / (2.*PETSC_PI))*(0.5 - x[0])*PetscSinReal(2.*PETSC_PI*x[0]/eps)
    + PetscSqr(eps / (2.*PETSC_PI))*(1. - PetscCosReal(2.*PETSC_PI*x[0]/eps));
  return 0;
}

/* integrand for the test function term */
static void f0_oscillatory_u
(PetscInt dim, PetscInt Nf, PetscInt NfAux,
 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
 const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
 const PetscScalar a_x[], PetscReal t, const PetscReal x[],
 PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -1.;
}

/* integrand for the test function gradient term */
static void f1_oscillatory_u
(PetscInt dim, PetscInt Nf, PetscInt NfAux,
 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
 const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
 const PetscScalar a_x[], PetscReal t, const PetscReal x[],
 PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal eps = 0.03125;

  f1[0] = u_x[0] / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

/* integrand for the test function gradient and basis function gradient term */
static void g3_oscillatory_uu
(PetscInt dim, PetscInt Nf, PetscInt NfAux,
 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
 const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
 const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
 PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal eps = 0.03125;

  g3[0] = 1. / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

static PetscErrorCode SetupMesh(const char prefix[], DM *dm) {

  PetscFunctionBeginUser;
  PetscCall(DMCreate(MPI_COMM_WORLD, dm));
  PetscCall(DMSetOptionsPrefix(*dm, prefix));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupSNES(const char prefix[], DM *dm, SNES *snes) {

  PetscFunctionBeginUser;
  PetscCall(SNESCreate(PETSC_COMM_WORLD, snes));
  PetscCall(SNESSetDM(*snes, *dm));
  PetscCall(SNESSetOptionsPrefix(*snes, prefix));
  PetscCall(SNESSetFromOptions(*snes));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM *dm) {
  PetscBool simplex;     /* is the first cell in dmplex a simplex */
  PetscInt  dim;         /* dimension of the dm */
  PetscFE   fe;          /* manages finite element space */

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(*dm, &dim));
  PetscCall(DMPlexIsSimplex(*dm, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(*dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(*dm));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM *dm, PetscDS *ds) {
  DMLabel         label;  /* label (?) */
  const PetscInt  id = 1; /* values for constrained points */
  
  PetscFunctionBeginUser;
  PetscCall(DMGetDS(*dm, ds));
  PetscCall(PetscDSSetResidual(*ds, 0, f0_oscillatory_u, f1_oscillatory_u));
  PetscCall(PetscDSSetJacobian(*ds, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu));
  PetscCall(PetscDSSetExactSolution(*ds, 0, oscillatory_u, NULL));
  PetscCall(DMGetLabel(*dm, "marker", &label));
  PetscCall(DMAddBoundary(*dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0,
			  NULL, (void (*)(void)) oscillatory_u, NULL, NULL, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SolveProblem(DM *dm, SNES *snes, Vec *u) {
  
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(*dm, u));
  PetscCall(VecSet(*u, 0.0));
  PetscCall(DMPlexSetSNESLocalFEM(*dm, NULL, NULL, NULL));
  PetscCall(SNESSolve(*snes, NULL, *u));
  PetscCall(SNESGetSolution(*snes, u));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  DM                    dmc, dmf;       /* problem specification coarse, fine */
  PetscDS               dsc, dsf;       /* discrete system coarse, fine */
  SNES                  snesc, snesf;   /* nonlinear solver coarse, fine */
  Vec                   uc, uf;         /* solution coarse, fine  */
  Vec                   exactc, exactf; /* exact soln coarse, fine */
  Vec                   luc, luf;       /* local soln coarse, fine */
  PetscViewer           viewer;         /* for viewing things */
  

  /* initialization and setup */
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(SetupMesh("coarse_", &dmc));
  PetscCall(SetupMesh("fine_", &dmf));
  PetscCall(SetupSNES("coarse_", &dmc, &snesc));
  PetscCall(SetupSNES("fine_", &dmf, &snesf));
  PetscCall(SetupDiscretization(&dmc));
  PetscCall(SetupDiscretization(&dmf));
  PetscCall(SetupProblem(&dmc, &dsc));
  PetscCall(SetupProblem(&dmf, &dsf));

  /* solution */
  PetscCall(SolveProblem(&dmc, &snesc, &uc));
  PetscCall(SolveProblem(&dmf, &snesf, &uf));

  /* view solution */
  PetscCall(VecViewFromOptions(uc, NULL, "-uc_view"));
  PetscCall(VecViewFromOptions(uf, NULL, "-uf_view"));

  /* setup viewer */
  PetscCall(PetscViewerCreate(MPI_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERDRAW));

  /* compare solutions coarse */ 
  PetscCall(DMGetGlobalVector(dmc, &exactc));
  PetscCall(DMGetLocalVector(dmc, &luc));
  PetscCall(DMComputeExactSolution(dmc, 0.0, exactc, NULL));
  PetscCall(DMGlobalToLocalBegin(dmc, uc, INSERT_VALUES, luc));
  PetscCall(DMGlobalToLocalEnd(dmc, uc, INSERT_VALUES, luc));
  PetscCall(DMPlexInsertBoundaryValues(dmc, PETSC_TRUE, luc, 0., NULL, NULL, NULL));
  PetscCall(DMPlexVecView1D(dmc, 1, &luc, viewer));
  PetscCall(DMRestoreGlobalVector(dmc, &exactc));

  /* compare solutions fine */
  PetscCall(DMGetGlobalVector(dmf, &exactf));
  PetscCall(DMGetLocalVector(dmf, &luf));
  PetscCall(DMComputeExactSolution(dmf, 0.0, exactf, NULL));
  PetscCall(DMGlobalToLocalBegin(dmf, uf, INSERT_VALUES, luf));
  PetscCall(DMGlobalToLocalEnd(dmf, uf, INSERT_VALUES, luf));
  PetscCall(DMPlexInsertBoundaryValues(dmf, PETSC_TRUE, luf, 0., NULL, NULL, NULL));
  PetscCall(DMPlexVecView1D(dmf, 1, &luf, viewer));
  PetscCall(DMRestoreGlobalVector(dmf, &exactf));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dmc));
  PetscCall(DMDestroy(&dmf));
  PetscCall(SNESDestroy(&snesc));
  PetscCall(SNESDestroy(&snesf));
  PetscCall(VecDestroy(&uc));
  PetscCall(VecDestroy(&uf));
  PetscCall(PetscFinalize());
  
  return 0;
}

