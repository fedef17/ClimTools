MODULE CLUSTER_TOOLKIT
!*****************************************************************************
! Routines for performing k-means cluster analysis  in PC space.
!
! Revision History:
!        Date        Programmer                Description
!        ====        ==========                ===========
!        2007        Franco Molteni (ECMWF)    Original code.
!
!      26/05/2011    Andrew Dawson (Oxford)    Updated to safe Fortran 90,
!                                              all subroutines have full
!                                              interfaces and support Python
!                                              via f2py.
!      
! Notes:
!     To compile with f2py:
!         f2py --fcompiler=gfortran --f90flags="-cpp"\
!                 -c -m <NAME> cluster_toolkit.F90 \
!                 only: clus_opt adran1 gausts tsstat
!
!*****************************************************************************


    IMPLICIT NONE
    

    PRIVATE
    PUBLIC :: ADRAN1, SEED_ADRAN1, CLUS_OPT, GAUSTS, TSSTAT, CLUS_SIG


    ! Store the current random number generator persistent variables.
    INTEGER, PARAMETER                   :: CT_ADRAN1_IR_LEN = 97
    INTEGER, DIMENSION(CT_ADRAN1_IR_LEN) :: CT_ADRAN1_IR     = 0
    INTEGER                              :: CT_ADRAN1_IY     = -1
    INTEGER                              :: CT_ADRAN1_SEED   = -11111


    CONTAINS


!*****************************************************************************
    SUBROUTINE CLUS_SIG (NRSAMP, PSIZE, NPART, NFLD, NPC, NDIS, PC,          &
            VAROPT, SIGNIFICANCE)
    !-------------------------------------------------------------------------
    ! Compute the significance of clusters as a percentage.
    !
    ! The percentage significance represents the percentage of red-noise
    ! samples whose optimal variance ratio is less than that given.
    !
    ! It is assumes that the number of variance ratios passed in is the
    ! number of partitions to try and that the first partition is the k=2
    ! partition and so on e.g., if there are 5 variance ratios it is assumed
    ! that the first is for the 2-cluster and the last for the 6-cluster.
    !
    ! Arguments:
    ! NRSAMP          input [INTEGER]
    !                 Number of red-noise samples to produce.
    ! PSIZE           input [INTEGER] {f2py implied}
    !                 The number of partitions to test.
    ! NPART           input [INTEGER]
    !                 The number of attempts used when determining the
    !                 optimal partition.
    ! NFLD            input [INTEGER] {f2py implied}
    !                 Number of fields in the PCS (length of the PC time
    !                 series).
    ! NPC             input [INTEGER] {f2py implied}
    !                 Number of PCs used to compute clusters.
    ! NDIS            input [INTEGER]
    !                 Number of discontinuities in the PC time series. It is
    !                 assumed that these discontinuities equally divide the
    !                 time series.
    ! VAROPT          input [REAL(PSIZE)]
    !                 Variance ratios to test for significance.
    ! SIGNIFICANCE    output [REAL(PSIZE)]
    !                 Percentage significance of each variance ratio.
    !
    !------------------------------------------------------------------------

        IMPLICIT NONE

        ! Input arguments.
        INTEGER,                    INTENT(IN) :: NRSAMP
        INTEGER,                    INTENT(IN) :: PSIZE
        INTEGER,                    INTENT(IN) :: NPART
        INTEGER,                    INTENT(IN) :: NFLD
        INTEGER,                    INTENT(IN) :: NPC
        INTEGER,                    INTENT(IN) :: NDIS
        REAL, DIMENSION(NPC, NFLD), INTENT(IN) :: PC
        REAL, DIMENSION(PSIZE),     INTENT(IN) :: VAROPT

        ! Declarations recognized by f2py.
        !f2py intent(hide) NFLD
        !f2py intent(hide) NPC
        !f2py intent(hide) PSIZE

        ! Output variables.
        REAL, DIMENSION(PSIZE), INTENT(OUT) :: SIGNIFICANCE

        ! Local variables.
        INTEGER                            :: JRS, JPC, NCL, NREM, IDX, NFLD1
        INTEGER, DIMENSION(PSIZE+1)        :: NFCL
        INTEGER, DIMENSION(NFLD)           :: INDCL
        INTEGER, DIMENSION(PSIZE+1, NPART) :: ISEED

        REAL                           :: TSM, INCREMENT
        REAL, DIMENSION(:), ALLOCATABLE:: TS
        REAL, DIMENSION(NPC)           :: PCSD, PCAC, PCM
        REAL :: STAT2
        REAL, DIMENSION(NPC, NFLD)     :: DPC
        REAL, DIMENSION(NPC, PSIZE+1)  :: CENTR
        
        ! Array of random number generators.
!        TYPE(ADRNG_T), DIMENSION(NRSAMP) :: RNGS
        !TYPE(RNG_T), DIMENSION(NRSAMP) :: RNGS

        ! Debugging.
        CHARACTER(LEN=15) :: DEBUG_ENV_NAME = "CTP_DEBUG_LEVEL"
        CHARACTER(LEN=3)  :: DEBUG_ENV_VALUE
        INTEGER           :: DEBUG_ENV_STATUS
        LOGICAL           :: DEBUG_INFO

        ! OpenMP variables.
!        INTEGER, PARAMETER :: CHUNKSIZE = 10
!        INTEGER :: CHUNK
!        INTEGER :: TID, NTHREADS
!        CHUNK = CHUNKSIZE

        ! Turn debugging on or off.
        CALL GET_ENVIRONMENT_VARIABLE (DEBUG_ENV_NAME, VALUE=DEBUG_ENV_VALUE,&
                STATUS=DEBUG_ENV_STATUS)
        IF (DEBUG_ENV_STATUS .NE. 0) THEN
            DEBUG_ENV_VALUE = "0"
        END IF
        IF (DEBUG_ENV_VALUE .EQ. "0") THEN
            DEBUG_INFO = .FALSE.
        ELSE
            DEBUG_INFO = .TRUE.
        END IF

        ALLOCATE(TS(NFLD))
        ! Compute PC statistics.
        DO JPC = 1, NPC
            TS(1:NFLD) = PC(JPC, 1:NFLD)
            CALL TSSTAT (TS, NFLD, NDIS, PCM(JPC), PCSD(JPC), PCAC(JPC))
        END DO

        ! Determine if the length of the PC time series is odd or even so
        ! this can be accounted for when constructing red-noise dummy PCs.
        NREM = MOD(NFLD, 2)
        NFLD1 = NFLD + NREM
        DEALLOCATE(TS)
        ALLOCATE(TS(NFLD1))

        INCREMENT    = 100. / REAL(MAX(1, NRSAMP))
        SIGNIFICANCE = 0.

        ! Loop over the requested number of samples, creating red-noise
        ! time series and performing the cluster analysis on them to
        ! obtain the optimal variance ratio for each.
        DO JRS = 1, NRSAMP

            IF (DEBUG_INFO) THEN
                WRITE (*, '("F90: COMPUTING SAMPLE ", I4)') JRS
            END IF

            ! Compute red-noise PC time series.
            DO JPC = 1, NPC
                CALL GAUSTS (NFLD1, 0., PCSD(JPC), PCAC(JPC),   &
                        NDIS, TS)
                TSM = SUM(TS(1:NFLD)) / REAL(NFLD)
                DPC(JPC, :) = TS(1:NFLD) - TSM
            END DO

            ! Compute the clusters from the red-noise sample time series.
            DO NCL = 2, PSIZE+1
                IDX = NCL - 1
                CALL CLUS_OPT (NFLD, NPC, NCL, NPART, DPC, NFCL,&
                        INDCL, CENTR, STAT2, ISEED) !(IDX, JRS)
                IF (VAROPT(IDX) .GT. STAT2) THEN
                    SIGNIFICANCE(IDX) = SIGNIFICANCE(IDX) + INCREMENT
                END IF
            END DO
        END DO

    END SUBROUTINE CLUS_SIG


!*****************************************************************************
    SUBROUTINE CLUS_OPT (NF, NPC, NCL, NPART, PC, NFCL, INDCL, CENTR,        &
            VAROPT, ISEED)
    !-------------------------------------------------------------------------
    ! Compute the optimal cluster partition for a given number of clusters.
    !
    !-------------------------------------------------------------------------
        
        IMPLICIT NONE

        ! Input variables.
        INTEGER,                  INTENT(IN) :: NF
        INTEGER,                  INTENT(IN) :: NPC
        INTEGER,                  INTENT(IN) :: NCL
        INTEGER,                  INTENT(IN) :: NPART
        REAL, DIMENSION(NPC, NF), INTENT(IN) :: PC
        !INTEGER,                  INTENT(IN) :: ISEED0
        !f2py intent(hide) NF
        !f2py intent(hide) NPC

        ! Output variables.
        INTEGER, DIMENSION(NF),         INTENT(OUT) :: INDCL
        INTEGER, DIMENSION(NCL),        INTENT(OUT) :: NFCL
        REAL,                           INTENT(OUT) :: VAROPT
        REAL, DIMENSION(NPC, NCL),      INTENT(OUT) :: CENTR
        INTEGER, DIMENSION(NCL, NPART), INTENT(OUT) :: ISEED

        ! Local variables.
        INTEGER :: MAXITER, JCL, JOPT, JPART
        REAL    :: RSEED, DSEED, VARTOT, D2SEED, R2SEED, VARINT

        ! Initialize local variables.
        MAXITER = 100
        RSEED   = 1.5
        DSEED   = 0.5

        VARTOT = SUM(PC ** 2) / REAL(NF)
        R2SEED = RSEED * RSEED * VARTOT
        D2SEED = DSEED * DSEED * VARTOT

        ! Loop over the number of partitions to evaluate. This component of
        ! the algorithm might be worth parallelizing if possible?
        JOPT = 0
        DO JPART = 1, NPART

            ! Select seeds.
            CALL SEL_SEED (NF, NPC, NCL, PC, R2SEED, D2SEED,          &
                    ISEED(1, jpart))

            ! Compute the cluster partition.
            CALL CLUS_PART (NF, NPC, NCL, PC, ISEED(1, JPART), MAXITER,       &
                    INDCL, NFCL, VARINT, CENTR)

            ! Work out if this is currently the most optimal partition.
            IF ((JPART .EQ. 1) .OR. (VARINT .LT. VAROPT)) THEN
                JOPT   = JPART
                VAROPT = VARINT
            END IF

        END DO

        ! Re-compute the optimal cluster partition. The indices and frequencies
        ! will have been lost computing subsequent non-optimal partitions.
        CALL CLUS_PART (NF, NPC, NCL, PC, ISEED(1, JOPT), MAXITER,           &
                INDCL, NFCL, VARINT, CENTR)

        ! Compute the variance ratio.
        VAROPT = 0
        DO JCL = 1, NCL
            VAROPT = VAROPT + NFCL(JCL) * SUM(CENTR(:, JCL) ** 2)
        END DO

        VAROPT = VAROPT / VARINT

    END SUBROUTINE CLUS_OPT


!*****************************************************************************
    SUBROUTINE CLUS_PART (NF, NPC, NCL, PC, ISEED, MAXITER, INDCL, NFCL,     &
            VARINT, CENTR)
    !-------------------------------------------------------------------------
    ! Compute a cluster partition in PC space for a given number of clusters.
    !
    !-------------------------------------------------------------------------

        IMPLICIT NONE

        INTEGER,                  INTENT(IN) :: NF
        INTEGER,                  INTENT(IN) :: NPC
        INTEGER,                  INTENT(IN) :: NCL
        REAL, DIMENSION(NPC, NF), INTENT(IN) :: PC
        INTEGER, DIMENSION(NCL),  INTENT(IN) :: ISEED
        INTEGER,                  INTENT(IN) :: MAXITER

        INTEGER, DIMENSION(NF),    INTENT(OUT) :: INDCL
        INTEGER, DIMENSION(NCL),   INTENT(OUT) :: NFCL
        REAL,                      INTENT(OUT) :: VARINT
        REAL, DIMENSION(NPC, NCL), INTENT(OUT) :: CENTR

        REAL :: D2, D2MIN
        INTEGER :: JMIN, ITER, JCL, JF, NCHANGE

        ! Initialize the cluster indices to an impossible value (cluster
        ! indices are defined to start at 1 in this code).
        INDCL = 0

        ! Initialize the centroids in PC space to the seed values.
        DO JCL = 1, NCL
            CENTR(:, JCL) = PC(:, ISEED(JCL))
        END DO

        ! Iterate the algorithm but only up until a given number of
        ! iterations.
        DO ITER = 1, MAXITER

            ! Initialize the number of cluster assignment changes to zero.
            NCHANGE = 0

            ! Loop over each point in time in the PC time series.
            DO JF = 1, NF

                ! Compute the distance of this point from the centroid of the
                ! first cluster.
                JMIN = 1
                D2MIN = SUM((PC(:, JF) - CENTR(:, 1)) ** 2)

                ! Loop over the remaining clusters.
                DO JCL = 2, NCL

                    ! Compute the distance from subsequent clusters
                    D2 = SUM((PC(:, JF) - CENTR(:, JCL)) ** 2)

                    ! If the distance is less than those before it store this
                    ! cluster as the closest.
                    IF (D2 .LT. D2MIN) THEN
                        JMIN = JCL
                        D2MIN = D2
                    END IF

                END DO

                ! If the cluster this point in PC space is associated has
                ! changed since the last iteration, update the record of
                ! changed assignments.
                IF (JMIN .NE. INDCL(JF)) THEN
                    NCHANGE = NCHANGE + 1
                END IF

                ! Record the new cluster assignment for this point in PC
                ! space.
                INDCL(JF) = JMIN

            END DO

            ! Break out of the iteration loop if no cluster assignments were
            ! changed during this iteration.
            IF (NCHANGE .EQ. 0) THEN
                EXIT
            END IF

            ! Set the frequency and centroid variables to zero.
            NFCL = 0
            CENTR = 0.

            ! Loop over all time points in the PC time series.
            DO JF = 1, NF
                ! Get the cluster number this point in PC space is assigned
                ! to.
                JCL = INDCL(JF)
                ! Increment the frequency counter for this cluster.
                NFCL(JCL) = NFCL(JCL) + 1
                ! Add the PC coordinates to the sum (which is later averaged
                ! to provide the cluster centroid.
                CENTR(:, JCL) = CENTR(:, JCL) + PC(:, JF)
            END DO

            ! Compute the cluster centroids.
            DO JCL = 1, NCL
                CENTR(:, JCL) = CENTR(:, JCL) / REAL(NFCL(JCL))
            END DO

        END DO

        ! Compute the internal variance.
        VARINT = 0.
        DO JF = 1, NF
            JCL = INDCL(JF)
            VARINT = VARINT + SUM((PC(:, JF) - CENTR(:, JCL)) ** 2)
        END DO

        RETURN

    END SUBROUTINE CLUS_PART


!*****************************************************************************
    SUBROUTINE SEL_SEED (NF, NPC, NCL, PC, R2SEED, D2SEED, ISEED)
    !------------------------------------------------------------------------
    ! Select a set of N seeds for the computation of an N-cluster partition.
    ! Seeds satisfy the following conditions:
    !   a) have a norm less than a maximum value in full PC space;
    !   b) have a minimum distance from eachother in full PC space;
    !   c) belong to different sectors of the PC1-PC2 plane.
    !
    ! This routine contains a GOTO to break out of a nested loop. Whilst not
    ! ideal this does make sense in the context of the algorithm and is
    ! difficult to replace (e.g., with a DO WHILE loop).
    !
    !------------------------------------------------------------------------

        IMPLICIT NONE

        ! Input variables.
        INTEGER,                  INTENT(IN) :: NF
        INTEGER,                  INTENT(IN) :: NPC
        INTEGER,                  INTENT(IN) :: NCL
        REAL, DIMENSION(NPC, NF), INTENT(IN) :: PC
        REAL,                     INTENT(IN) :: R2SEED
        REAL,                     INTENT(IN) :: D2SEED

        ! Output variables.
        INTEGER, DIMENSION(NCL), INTENT(OUT) :: ISEED

        ! Local variables.
        INTEGER :: JSEED, JCL, JTRY, KCL, KSEED
        REAL :: COSPN, RNORM, COSTF, COST1, COSTMAX, D2, DOTP, R2J, R2K

        COSPN = COS(2. * ASIN(1.) / REAL(NCL))

        JSEED = 0
        RNORM = R2SEED + 1.
        DO WHILE (RNORM .GT. R2SEED)
            JSEED = 1 + INT(NF * ADRAN1())
            RNORM = SUM(PC(:, JSEED) ** 2)
        END DO

        ISEED(1) = JSEED

        DO JCL = 1, NCL

            COSTMAX = 0.
            COST1 = 1. / (JCL - 1.)

            DO JTRY = 1, 10

200             JSEED = 1 + INT(NF * ADRAN1())
                COSTF = 0.001

                RNORM = SUM(PC(:, JSEED) **2)
                IF (RNORM .LE. R2SEED) THEN
                    COSTF = COSTF + 1.
                END IF
                
                R2J = PC(1, JSEED) * PC(1, JSEED) + PC(2, JSEED) * PC(2, JSEED)

                DO KCL = 1, JCL - 1

                    KSEED = ISEED(KCL)

                    IF (JSEED .EQ. KSEED) THEN
                        GOTO 200
                    END IF

                    D2 = SUM((PC(:, JSEED) - PC(:, KSEED)) ** 2)
                    IF (D2 .GT. D2SEED) THEN
                        COSTF = COSTF + COST1
                    END IF

                    DOTP = PC(1, JSEED) * PC(1, KSEED) +                     &
                            PC(2, JSEED) * PC(2, KSEED)
                    
                    R2K = PC(1, KSEED) * PC(1, KSEED) +                      &
                            PC(2, KSEED) * PC(2, KSEED)

                    IF (DOTP .LT. COSPN * SQRT(R2J * R2K)) THEN
                        COSTF = COSTF + COST1
                    END IF

                END DO

                IF (COSTF .GT. COSTMAX) THEN
                    ISEED(JCL) = JSEED
                    COSTMAX = COSTF
                END IF

            END DO

        END DO

    END SUBROUTINE SEL_SEED


!*****************************************************************************
    SUBROUTINE SEED_ADRAN1 (SEED)
    !------------------------------------------------------------------------
    ! Set the seed used for ADRAN1. To re-initialize the generator this
    ! should be set to a negative number.
    !
    ! This is just a functional way of setting the module variable
    ! CT_ADRAN1_SEED.
    !
    !------------------------------------------------------------------------

        IMPLICIT NONE

        ! Input variables.
        INTEGER, INTENT(IN) :: SEED

        ! Set the module variable for the seed.
        CT_ADRAN1_SEED = SEED

    END SUBROUTINE SEED_ADRAN1


!*****************************************************************************
    FUNCTION ADRAN1 ()
    !------------------------------------------------------------------------
    ! Return a uniform random deviate between 0.0 and 1.0.
    !
    ! Based on numerical recipes in Fortran chapter 7.1.
    !
    ! The state of the generator is stored in the module variables prefixed
    ! with CT_ADRAN1.
    !
    !------------------------------------------------------------------------

        IMPLICIT NONE

        ! Return variable.
        REAL :: ADRAN1

        ! Local parameters.
        INTEGER, PARAMETER :: IM = 714025
        INTEGER, PARAMETER :: IA = 1366
        INTEGER, PARAMETER :: IC = 150889
        REAL,    PARAMETER :: RM = 1. / IM

        ! Local variables.
        INTEGER :: J

        ! Initialize the generator if the seed is negative or it is presently
        ! uninitialized.
        IF (CT_ADRAN1_IY .LT. 0) THEN

            CT_ADRAN1_SEED = MOD(IC + ABS(CT_ADRAN1_SEED), IM)

            DO J = 1, CT_ADRAN1_IR_LEN
              
                CT_ADRAN1_SEED = MOD(IA * CT_ADRAN1_SEED + IC, IM)

                CT_ADRAN1_IR(J) = CT_ADRAN1_SEED

            END DO

            CT_ADRAN1_SEED = MOD(IA * CT_ADRAN1_SEED + IC, IM)

            CT_ADRAN1_IY = CT_ADRAN1_SEED

        END IF

        J = 1 + (CT_ADRAN1_IR_LEN * CT_ADRAN1_IY) / IM

        CT_ADRAN1_IY = CT_ADRAN1_IR(J)

        ADRAN1 = CT_ADRAN1_IY * RM

        CT_ADRAN1_SEED = MOD(IA * CT_ADRAN1_SEED + IC, IM)

        CT_ADRAN1_IR(J) = CT_ADRAN1_SEED
         
    END FUNCTION ADRAN1
      

!*****************************************************************************
    SUBROUTINE GAUSTS (NT, AV, SD, AC, NDIS, TS)
    !------------------------------------------------------------------------
    ! Compute a Gaussian distributed time series (TS) of (NT) random values,
    ! with assigned average (AV), standard deviation (SD), and lag-1
    ! auto-correlation (AC).
    !
    ! Auto-correlation may be discontinued at the limits between (NDIS)
    ! sub-series.
    !
    !------------------------------------------------------------------------
        
        IMPLICIT NONE
        
        !Input variables.
        INTEGER, INTENT(IN) :: NT
        REAL,    INTENT(IN) :: AV
        REAL,    INTENT(IN) :: SD
        REAL,    INTENT(IN) :: AC
        INTEGER, INTENT(IN) :: NDIS

        ! Output variables.
        REAL, DIMENSION(NT), INTENT(OUT) :: TS

        ! Local variables.
        INTEGER :: J, J2, NT2, JD
        REAL    :: V1, V2, RSQ, FACT, SD2

        ! Generate a time series on NT Gaussian deviates.
        DO J = 2, NT, 2

            RSQ = 1000.
            DO WHILE ((RSQ .GT. 1) .OR. (RSQ .EQ. 0))
                V1 = 2. * ADRAN1() - 1
                V2 = 2. * ADRAN1() - 1
                RSQ = V1 * V1 + V2 * V2
            END DO

            FACT = SQRT(-2. * LOG(RSQ) / RSQ)
            TS(J-1) = V1 * FACT
            TS(J)   = V2 * FACT

        END DO

        ! Introduce auto-correlation.
        IF (AC .NE. 0) THEN

            NT2 = NT / MAX(1, NDIS)
            SD2 = SQRT(1. - AC * AC)

            J = 0
            DO JD = 1, NDIS
                
                J = J + 1

                DO J2 = 2, NT2

                    J = J + 1
                    TS(J) = AC * TS(J-1) + SD2 * TS(J)

                END DO

            END DO

        END IF

        ! Set assigned average and standard deviation.
        DO J = 1, NT
            TS(J) = SD * TS(J) + AV
        END DO

    END SUBROUTINE GAUSTS


!*****************************************************************************
    SUBROUTINE TSSTAT (TS, NT, NDIS, AV, SD, AC)

        IMPLICIT NONE
        
        ! Input variables.
        INTEGER,             INTENT(IN) :: NT
        REAL, DIMENSION(NT), INTENT(IN) :: TS
        INTEGER,             INTENT(IN) :: NDIS

        ! Declarations recognized by f2py.
        !f2py intent(hide) nt

        ! Output variables.
        REAL, INTENT(OUT) :: AV
        REAL, INTENT(OUT) :: SD
        REAL, INTENT(OUT) :: AC

        ! Local variables.
        INTEGER :: NT2, J, JD, J2
        REAL    :: VAR, COV, DEV1, DEV2

        NT2 = NT / MAX(1, NDIS)

        AV = 0.
        DO J = 1, NT
            AV = AV + TS(J)
        END DO
        AV = AV / REAL(NT)

        VAR = 0.
        COV = 0.

        J = 0
        DO JD = 1, NDIS
            DEV1 = 0.
            DO J2 = 1, NT2
                J = J + 1
                DEV2 = TS(J) - AV
                VAR = VAR + DEV2 * DEV2
                COV = COV + DEV1 * DEV2
                DEV1 = DEV2
            END DO
        END DO

        SD = SQRT(VAR / REAL(NT))
        AC = (COV * NT) / (VAR * (NT - NDIS))

    END SUBROUTINE TSSTAT


END MODULE CLUSTER_TOOLKIT
