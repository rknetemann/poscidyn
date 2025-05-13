!----------------------------------------------------------------------
!  oscillators.f90  –  N-DOF nonlinear forced oscillators
!                     q̈ + c q̇ + k q + α q² + γ q³ = f cos(ω t)
!                     turned into 1st order and made autonomous
!                     with the harmonic pair (y1,y2) = (cos ωt, sin ωt)
!
!  NDIM = 2*N + 2
!         ├─ q₁…qN       (positions)
!         ├─ v₁…vN       (velocities)
!         └─ y1, y2      (harmonic driver variables)
!----------------------------------------------------------------------

      SUBROUTINE FUNC (NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
      IMPLICIT NONE
      INTEGER NDIM, IJAC, ICP(*)
      DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)

!*--- bookkeeping
      INTEGER, PARAMETER          :: MAXN = 10
      INTEGER                     :: N, i, j, k, l, idx
      DOUBLE PRECISION            :: mass (MAXN), damp (MAXN), forc (MAXN)
      DOUBLE PRECISION            :: kk (MAXN,MAXN)
      DOUBLE PRECISION            :: aa (MAXN,MAXN,MAXN)
      DOUBLE PRECISION            :: gg (MAXN,MAXN,MAXN,MAXN)
      DOUBLE PRECISION            :: q  (MAXN), v  (MAXN)
      DOUBLE PRECISION            :: y1, y2, omega
      DOUBLE PRECISION            :: s_lin, s_quad, s_cub

!*--- how many oscillators?
      N = (NDIM-2)/2

!*--- unpack parameters -----------------------------------------------
      idx = 1
! masses
      DO i = 1, N
         mass(i) = PAR(idx);  idx = idx + 1
      END DO
! damping
      DO i = 1, N
         damp(i) = PAR(idx);  idx = idx + 1
      END DO
! linear stiffness
      DO i = 1, N
         DO j = 1, N
            kk(i,j) = PAR(idx); idx = idx + 1
         END DO
      END DO
! quadratic coupling
      DO i = 1, N
         DO j = 1, N
            DO k = 1, N
               aa(i,j,k) = PAR(idx); idx = idx + 1
            END DO
         END DO
      END DO
! cubic coupling
      DO i = 1, N
         DO j = 1, N
            DO k = 1, N
               DO l = 1, N
                  gg(i,j,k,l) = PAR(idx); idx = idx + 1
               END DO
            END DO
         END DO
      END DO
! forcing amplitudes  fᵢ
      DO i = 1, N
         forc(i) = PAR(idx);  idx = idx + 1
      END DO
! forcing frequency  ω
      omega = PAR(idx)

!*--- state variables --------------------------------------------------
      DO i = 1, N
         q(i) = U(i)
         v(i) = U(N+i)
      END DO
      y1 = U(2*N+1)
      y2 = U(2*N+2)

!*--- equations --------------------------------------------------------
! q̇ᵢ = vᵢ
      DO i = 1, N
         F(i) = v(i)
      END DO

! v̇ᵢ = ( -c v  -Σk q -Σα q² -Σγ q³ + fᵢ · y1 ) / mᵢ
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

         F(N+i) = (-damp(i)*v(i) - s_lin - s_quad - s_cub +            &
     &             forc(i)*y1) / mass(i)
      END DO

! harmonic driver:  ẏ1 = -ω y2 ,   ẏ2 = ω y1
      F(2*N+1) = -omega * y2
      F(2*N+2) =  omega * y1

      RETURN
      END SUBROUTINE FUNC
!----------------------------------------------------------------------


      SUBROUTINE STPNT (NDIM,U,PAR,T)
      IMPLICIT NONE
      INTEGER NDIM, idx, i, j, k, l, N
      DOUBLE PRECISION U(NDIM), PAR(*), T
      INTEGER, PARAMETER :: MAXN = 10

      N   = (NDIM-2)/2
      idx = 1

! masses
      DO i = 1, N
         PAR(idx) = 1.d0 ; idx = idx + 1
      END DO
! damping
      DO i = 1, N
         PAR(idx) = 0.05d0 ; idx = idx + 1
      END DO
! linear k = I
      DO i = 1, N
         DO j = 1, N
            PAR(idx) = MERGE(1.d0, 0.d0, i == j)
            idx = idx + 1
         END DO
      END DO
! quadratic α = 0
      DO i = 1, N*N*N
         PAR(idx) = 0.d0 ; idx = idx + 1
      END DO
! cubic γ = 0
      DO i = 1, N*N*N*N
         PAR(idx) = 0.d0 ; idx = idx + 1
      END DO
! forcing amplitudes  fᵢ = 0   (we’ll increase later)
      DO i = 1, N
         PAR(idx) = 0.d0 ; idx = idx + 1
      END DO
! forcing frequency  ω = 1
      PAR(idx) = 1.d0

! initial state  (rest, but y1 = 1, y2 = 0)
      DO i = 1, NDIM
         U(i) = 0.d0
      END DO
      U(2*N+1) = 1.d0     ! y1
      U(2*N+2) = 0.d0     ! y2

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
