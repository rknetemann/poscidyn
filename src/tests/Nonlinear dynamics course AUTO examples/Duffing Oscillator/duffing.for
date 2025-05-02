C----------------------------------------------------------------------
C----------------------------------------------------------------------
C  DUFFING 
C----------------------------------------------------------------------
C----------------------------------------------------------------------
C
      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
C     ---------- ----
C
C This subroutine evaluates the right hand side of the first order
C system and the derivatives with respect to (U(1),U(2))
C and with respect to the free parameters.
C
C Input parameters :
C	NDIM   -   Dimension of U and F.
C	U      -   Vector containing U.
C	PAR    -   Array of parameters in the differential equations.
C	ICP    -   PAR(ICP(1)) is the initial 'free' parameter.
C		   PAR(ICP(2)) is a secondary 'free' parameter,
C		   for subsequent 2-parameter continuations.
C	IJAC   -   =1 if the Jacobians DFDU and DFDP are to be returned,
C		   =0 if only F(U,PAR) is to be returned in this call.
C
C Values to be returned :
C	F     -   F(U,PAR)  the right hand side of the ODE.
C	DFDU  -   The derivative (Jacobian) with respect to U.
C		  DFDU(i,j) must be given the value of d F(i) / d U(j) .
C
C	DFDP  -   The derivative with respect to the 'free' parameters:
C		  DFDP(i,ICP(j)) =  d F(i) /d PAR(ICP(j)).
C
c	USE MSIMSL
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     SGLE IMPLICIT REAL (A-H,O-Z)
C
      DIMENSION U(NDIM),PAR(20)
      DIMENSION F(NDIM),DFDU(NDIM,NDIM),DFDP(NDIM,20)
	real*8 q,dq,k,k2,k3,P


	TPI=4*DATAN(1.0D0)



      ft1=PAR(1)/1000.d0
	omega=PAR(2)
	

   

      

	 q=u(1)
	 dq=u(2)
       
	 F(1)=U(2)
       F(2)=-q -30*q**3 -0.04*dq +ft1*U(3)

       F(3)=U(3)+omega*U(4)-U(3)*((U(3))**2+(U(4))**2)
	 F(4)=-omega*U(3)+U(4)-U(4)*((U(3))**2+(U(4))**2)


	




	IF(IJAC.EQ.0)RETURN


      RETURN
      END


C
      SUBROUTINE STPNT(NDIM,U,PAR,T)
C     ---------- -----
C
C In this subroutine the steady state starting point must be defined.
C (Used when not restarting from a previously computed solution).
C The problem parameters (PAR) may be initialized here or else in INIT.
C
C	NDIM   -   Dimension of the system of equations.
C	U      -   Vector of dimension NDIM.
C		   Upon return U should contain a steady state solution
C		   corresponding to the values assigned to PAR.
C	PAR    -   Array of parameters in the differential equations.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
      DIMENSION U(NDIM),PAR(20)
C
C Initialize the problem parameters.
       TPI=8*DATAN(1.0D0)

	 omega=0.2
	 ft1=0
	 
	 PAR(1)=ft1*1000.
	 PAR(2)=omega
	 

c	 PAR(5)=gi1/h
c	 PAR(6)=gi2/h
c	 PAR(7)=gi3/h
c	 PAR(8)=gi4/h
c	 PAR(9)=gi5/h
c	 PAR(10)=gi6/h
	 PAR(11)=TPI/omega

	do ii=1,2
	 U(ii)=0.
	enddo
	 U(3)=DSIN(TPI*T)
	 U(4)=DCOS(TPI*T)


      RETURN
      END


C
C

C The following subroutines are not used in this example
C
      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
	RETURN
      END
C
      SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
      RETURN
      END
C
      SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)
      RETURN
      END
C
      SUBROUTINE PVLS(NDM,U,PAR)
      RETURN
      END

