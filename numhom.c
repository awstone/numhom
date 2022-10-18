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
  Vec                   uc, uf;         /* solution coarse, fine */

  /* initialize petsc */
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  /* setup mesh coarse */
  PetscCall(SetupMesh("coarse_", &dmc));
  /* setup mesh fine */
  PetscCall(SetupMesh("fine_", &dmf));
  /* setup snes coarse */
  PetscCall(SetupSNES("coarse_", &dmc, &snesc));
  /* setup sens fine */
  PetscCall(SetupSNES("fine_", &dmf, &snesf));
  /* setup discretization coarse */
  PetscCall(SetupDiscretization(&dmc));
  /* setup discretization fine */
  PetscCall(SetupDiscretization(&dmf));
  /* setup problem coarse */
  PetscCall(SetupProblem(&dmc, &dsc));
  /* setup problem fine */
  PetscCall(SetupProblem(&dmf, &dsf));

  /* solve problem coarse */
  PetscCall(SolveProblem(&dmc, &snesc, &uc));
  /* PetscCall(DMCreateGlobalVector(dmc, &uc)); */
  /* PetscCall(VecSet(uc, 0.0)); */
  /* PetscCall(DMPlexSetSNESLocalFEM(dmc, NULL, NULL, NULL)); */
  /* PetscCall(SNESSolve(snesc, NULL, uc)); */
  /* //  PetscCall(SNESView(snesc, NULL)); */
  /* PetscCall(SNESGetSolution(snesc, &uc)); */

  /* solve problem fine */
  PetscCall(SolveProblem(&dmf, &snesf, &uf));
  /* PetscCall(DMCreateGlobalVector(dmf, &uf)); */
  /* PetscCall(VecSet(uf, 0.0)); */
  /* PetscCall(DMPlexSetSNESLocalFEM(dmf, NULL, NULL, NULL)); */
  /* PetscCall(SNESSolve(snesf, NULL, uf)); */
  /* //d  PetscCall(SNESView(snesf, NULL)); */
  /* PetscCall(SNESGetSolution(snesf, &uf)); */

  /* show solutions */
  PetscCall(VecViewFromOptions(uc, NULL, "-uc_view"));
  PetscCall(VecViewFromOptions(uf, NULL, "-uf_view"));

  /* clean up */
  PetscCall(DMDestroy(&dmc));
  PetscCall(DMDestroy(&dmf));
  PetscCall(SNESDestroy(&snesc));
  PetscCall(SNESDestroy(&snesf));
  PetscCall(VecDestroy(&uc));
  PetscCall(VecDestroy(&uf));
  PetscCall(PetscFinalize());
  
  return 0;
}

