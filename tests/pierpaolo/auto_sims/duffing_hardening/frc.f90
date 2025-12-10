!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   frc :      A periodically forced system
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!     ---------- ---- 

 
      IMPLICIT NONE

!------------------------------------------------------------------------
! AUTO interface
!------------------------------------------------------------------------
      INTEGER, INTENT(IN)    :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN)    :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT)   :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

!------------------------------------------------------------------------
! Local variables
!------------------------------------------------------------------------
      DOUBLE PRECISION q1,q2,q3,q4
      DOUBLE PRECISION z1,z2
      DOUBLE PRECISION fx, fy, omega
      DOUBLE PRECISION Qx, Qy, omega0x, omega0y,gamx,gamy,alpha



! ---------------------------------------------------------
! Parameters to be continued
! PAR(1) = fx   (drive amplitude on x)
! PAR(2) = fy   (drive amplitude on y)
! PAR(3) = omega (drive frequency)

       fx    = PAR(1)
       
       omega = PAR(2)


!------------------------------------------------------------------------
! **Fixed physical constants** from your model
!------------------------------------------------------------------------
       Qx     = 10.0

       omega0x = 1.0

       gamx   = 2.67e-2


       q1   = U(1)
       q2   = U(2)



! ---------------------------------------------------------
! Cubic nonlinear terms
       z1 = gamx*q1*q1*q1

! ---------------------------------------------------------
! ODEs for x and y
       F(1) = q2

       F(2) = - (omega0x/Qx)*q2 - omega0x**2*q1 - z1 + fx*U(5)


! ---------------------------------------------------------
! Driving oscillator producing cos(omega t), sin(omega t)
!   U(5) = cos(omega t)
!   U(6) = sin(omega t)

        F(5)=U(5)+omega*U(6)-U(5)*((U(5))**2+(U(6))**2)
       F(6)=-omega*U(5)+U(6)-U(6)*((U(5))**2+(U(6))**2)
 
      END SUBROUTINE FUNC

          SUBROUTINE STPNT(NDIM, U, PAR, T)
      IMPLICIT NONE

      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION TPI, omega, fx, fy

! ------------------------------------------------------------
! 2*pi
! ------------------------------------------------------------
       TPI=8*DATAN(1.0D0)

! ------------------------------------------------------------
! Initial parameter values
! ------------------------------------------------------------
      fx    = 0.03D0        ! forcing amplitude on x
      omega = 0.5D0       ! drive frequency

      PAR(1) = fx
     
      PAR(2) = omega

      PAR(11)=TPI/PAR(2)
! If you want period as a parameter, uncomment:
!     PAR(4) = TPI / omega

! ------------------------------------------------------------
! Initial state for x, x'
! ------------------------------------------------------------
      U(1) = 0.0 
      U(2) = 0.0  

! ------------------------------------------------------------
! Init for forcing oscillator:
!   U(5) = cos(ωt)
!   U(6) = sin(ωt)
! AUTO supplies T in [0,1], so argument = 2π T
! ------------------------------------------------------------
      U(5) = DSIN(TPI * T)
      U(6) = DCOS(TPI * T)

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
