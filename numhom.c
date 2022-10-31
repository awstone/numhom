#include "mpi.h"
#include "petscdm.h"
#include "petscmat.h"
#include "petscsystypes.h"
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

static PetscErrorCode DrawSolution(DM *dm, Vec *u) {
  PetscViewer     viewer;       /* viewer object */
  Vec             exact;        /* exact soln */
  Vec             ul, exactl;   /* local */
  Vec             vecs[2];      /* container for exact and ul */
  
  PetscFunctionBeginUser;
  /* setup viewer */
  PetscCall(PetscViewerCreate(MPI_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERDRAW));
  /* compute exact solution */
  PetscCall(DMGetGlobalVector(*dm, &exact));
  PetscCall(DMComputeExactSolution(*dm, 0.0, exact, NULL));
  /* add boundary conditions to solution */
  PetscCall(DMGetLocalVector(*dm, &ul));
  PetscCall(DMGlobalToLocalBegin(*dm, *u, INSERT_VALUES, ul));
  PetscCall(DMGlobalToLocalEnd(*dm, *u, INSERT_VALUES, ul));
  PetscCall(DMPlexInsertBoundaryValues(*dm, PETSC_TRUE, ul, 0., NULL, NULL, NULL));
  /* add boundary conditions to exact solution */
  PetscCall(DMGetLocalVector(*dm, &exactl));
  PetscCall(DMGlobalToLocalBegin(*dm, exact, INSERT_VALUES, exactl));
  PetscCall(DMGlobalToLocalEnd(*dm, exact, INSERT_VALUES, exactl));
  PetscCall(DMPlexInsertBoundaryValues(*dm, PETSC_TRUE, exactl, 0., NULL, NULL, NULL));
  vecs[0] = exactl;
  vecs[1] = ul;
  /* plot vectors */
  PetscCall(DMPlexVecView1D(*dm, 2, vecs, viewer));
  /* cleanup */
  PetscCall(DMRestoreLocalVector(*dm, &vecs[0]));
  PetscCall(DMRestoreLocalVector(*dm, &vecs[1]));
  PetscCall(DMRestoreGlobalVector(*dm, &exact));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  DM          dmc, dmf;       /* problem specification coarse, fine */
  PetscDS     dsc, dsf;       /* discrete system coarse, fine */
  SNES        snesc, snesf;   /* nonlinear solver coarse, fine */
  Vec         uc, uf;         /* solution coarse, fine  */
  Mat         mf, pf;         /* mass matrix, projection matrix fine */
  Mat         ac, af;         /* stiffness matrix coarse, fine */
  Mat         pa, papt;     /* matrix products PA, PAP^T */

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

  /* print solution */
  PetscCall(VecViewFromOptions(uc, NULL, "-uc_view"));
  PetscCall(VecViewFromOptions(uf, NULL, "-uf_view"));

  /* draw solution */
  PetscCall(DrawSolution(&dmc, &uc));
  PetscCall(DrawSolution(&dmf, &uf));

  /* create projection */
  PetscCall(DMCreateMassMatrix(dmc, dmf, &mf));
  PetscCall(DMCreateInterpolation(dmc, dmf, &pf, NULL));
  PetscCall(MatTranspose(pf, MAT_INPLACE_MATRIX, &pf));
  PetscCall(MatViewFromOptions(mf, NULL, "-mf_view"));
  PetscCall(MatViewFromOptions(pf, NULL, "-pf_view"));

  /* get stiffness matrix fine */
  PetscCall(DMCreateMatrix(dmf, &af));
  PetscCall(SNESSetJacobian(snesf, af, af, NULL, NULL));
  PetscCall(SNESComputeJacobian(snesf, uf, af, af));
  PetscCall(MatViewFromOptions(af, NULL, "-af_view"));

  /* get stiffness matrix coarse */
  PetscCall(DMCreateMatrix(dmc, &ac));
  PetscCall(SNESSetJacobian(snesc, ac, ac, NULL, NULL));
  PetscCall(SNESComputeJacobian(snesc, uc, ac, ac));
  PetscCall(MatViewFromOptions(ac, NULL, "-ac_view"));

  /* project to stiffness matrix coarse */
  PetscCall(MatMatMult(pf, af, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &pa));
  PetscCall(MatMatTransposeMult(pa, pf, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &papt));
  PetscCall(MatViewFromOptions(papt, NULL, "-papt_view"));
  
  /* cleanup */
  PetscCall(DMDestroy(&dmc));
  PetscCall(DMDestroy(&dmf));
  PetscCall(SNESDestroy(&snesc));
  PetscCall(SNESDestroy(&snesf));
  PetscCall(VecDestroy(&uc));
  PetscCall(VecDestroy(&uf));
  PetscCall(MatDestroy(&mf));
  PetscCall(MatDestroy(&pf));
  PetscCall(MatDestroy(&ac));
  PetscCall(MatDestroy(&af));
  PetscCall(MatDestroy(&pa));
  PetscCall(MatDestroy(&papt));
  PetscCall(PetscFinalize());
  
  return 0;
}


