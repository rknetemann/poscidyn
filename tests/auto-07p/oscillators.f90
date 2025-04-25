!----------------------------------------------------------------------
!  oscillators.f90  –  N-DOF nonlinear forced oscillators
!                     q̈ + c q̇ + k q + α q² + γ q³ = f cos(ω t)
!                     transformed to 1st order and made autonomous
!
!  NDIM = 2*N + 1  (q, v = q̇, ϕ)     i = 1 … N
!----------------------------------------------------------------------

      SUBROUTINE FUNC (NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     -----  ----
      IMPLICIT NONE
      INTEGER NDIM, IJAC, ICP(*)
      DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)

!*--- local bookkeeping
      INTEGER, PARAMETER      :: MAXN = 10          ! enlarge if required
      INTEGER                 :: N, i, j, k, l, idx
      DOUBLE PRECISION        :: mass (MAXN), damp (MAXN), forc (MAXN)
      DOUBLE PRECISION        :: kk (MAXN,MAXN)
      DOUBLE PRECISION        :: aa (MAXN,MAXN,MAXN)
      DOUBLE PRECISION        :: gg (MAXN,MAXN,MAXN,MAXN)
      DOUBLE PRECISION        :: q  (MAXN), v  (MAXN), phi, omega
      DOUBLE PRECISION        :: s_lin, s_quad, s_cub

!*--- how many oscillators?
      N = (NDIM-1)/2

!*--- unpack parameters ------------------------------------------------
      idx = 1
! masses mass(i)
      DO i = 1, N
         mass(i) = PAR(idx);  idx = idx + 1
      END DO
! damping damp(i)
      DO i = 1, N
         damp(i) = PAR(idx);  idx = idx + 1
      END DO
! stiffness k(i,j)
      DO i = 1, N
         DO j = 1, N
            kk(i,j) = PAR(idx); idx = idx + 1
         END DO
      END DO
! quadratic α(i,j,k)
      DO i = 1, N
         DO j = 1, N
            DO k = 1, N
               aa(i,j,k) = PAR(idx); idx = idx + 1
            END DO
         END DO
      END DO
! cubic γ(i,j,k,l)
      DO i = 1, N
         DO j = 1, N
            DO k = 1, N
               DO l = 1, N
                  gg(i,j,k,l) = PAR(idx); idx = idx + 1
               END DO
            END DO
         END DO
      END DO
! forcing amplitudes forc(i)
      DO i = 1, N
         forc(i) = PAR(idx);  idx = idx + 1
      END DO
! forcing frequency ω
      omega = PAR(idx)

!*--- state variables --------------------------------------------------
      DO i = 1, N
         q(i) = U(i)           ! positions
         v(i) = U(N+i)         ! velocities
      END DO
      phi = U(2*N+1)           ! phase variable

!*--- equations --------------------------------------------------------
! first N:  q̇ᵢ = vᵢ
      DO i = 1, N
         F(i) = v(i)
      END DO

! next N:  v̇ᵢ = (-cᵢ vᵢ - Σk q - Σα q² - Σγ q³ + fᵢ cosϕ) / mᵢ
      DO i = 1, N
         s_lin  = 0.d0
         s_quad = 0.d0
         s_cub  = 0.d0

         DO j = 1, N
            s_lin = s_lin + kk(i,j) * q(j)
            DO k = 1, N
               s_quad = s_quad + aa(i,j,k) * q(j) * q(k)
               DO l = 1, N
                  s_cub = s_cub + gg(i,j,k,l) * q(j) * q(k) * q(l)
               END DO
            END DO
         END DO

         F(N+i) = (-damp(i)*v(i) - s_lin - s_quad - s_cub +           &
     &             forc(i)*COS(phi)) / mass(i)
      END DO

! last one:  ϕ̇ = ω
      F(2*N+1) = omega

      RETURN
      END SUBROUTINE FUNC
!----------------------------------------------------------------------


      SUBROUTINE STPNT (NDIM,U,PAR,T)
!     ------- -----
      IMPLICIT NONE
      INTEGER NDIM, i, j, k, l, idx, N
      DOUBLE PRECISION U(NDIM), PAR(*), T
      INTEGER, PARAMETER :: MAXN = 10

      N = (NDIM-1)/2            ! keep consistent with FUNC
      idx = 1

! --- masses ----------------------------------------------------------
      DO i = 1, N
         PAR(idx) = 1.d0         ! mass(i)
         idx = idx + 1
      END DO
! --- damping ---------------------------------------------------------
      DO i = 1, N
         PAR(idx) = 0.05d0       ! damp(i)
         idx = idx + 1
      END DO
! --- stiffness (unit matrix) -----------------------------------------
      DO i = 1, N
         DO j = 1, N
            IF (i .EQ. j) THEN
               PAR(idx) = 1.d0   ! diagonal springs
            ELSE
               PAR(idx) = 0.d0
            END IF
            idx = idx + 1
         END DO
      END DO
! --- quadratic -------------------------------------------------------
      DO i = 1, N*N*N
         PAR(idx) = 0.d0
         idx = idx + 1
      END DO
! --- cubic -----------------------------------------------------------
      DO i = 1, N*N*N*N
         PAR(idx) = 0.d0
         idx = idx + 1
      END DO
! --- forcing amplitudes ---------------------------------------------
      DO i = 1, N
         PAR(idx) = 0.d0         ! forc(i)  (change later in continuation)
         idx = idx + 1
      END DO
! --- forcing frequency ----------------------------------------------
      PAR(idx) = 0.d0      ! ω  = 0 → equilibrium exists
      
! --- initial state ---------------------------------------------------
      DO i = 1, NDIM
         U(i) = 0.d0             ! all variables start at rest
      END DO

      END SUBROUTINE STPNT
!----------------------------------------------------------------------

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
!----------------------------------------------------------------------
