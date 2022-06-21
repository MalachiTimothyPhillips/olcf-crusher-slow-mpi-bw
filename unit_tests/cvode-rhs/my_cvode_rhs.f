c----------------------------------------------------------------------
      subroutine cvunpack(w1,y)
c
c     copy the internal cvode vector y to nek array w1 
c
      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      real w1(lx1,ly1,lz1,lelt,*)
      real y(*)

      nxyz = lx1*ly1*lz1

      j = 1
        do ifield = 2,nfield
           if (ifcvfld(ifield)) then
              ntot = nxyz*nelfld(ifield)
              call copy   (w1(1,1,1,1,ifield-1),y(j),ntot)
              call bcdirsc(w1(1,1,1,1,ifield-1)) ! restore dirichlet bcs
              j = j + ntot
           endif
        enddo

      return
      end
c----------------------------------------------------------------------
      subroutine cvpack(y,w1,ifrhs)
c
c     copy the nek array w1 to the internal cvode vector y
c     note: assumes temperature is stored in ifield=2 (only for ifdp0dt)
c
      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      real y(*)
      real w1(lx1,ly1,lz1,lelt,*)
      logical ifrhs

      common /scrsf/ dtmp(lx1,ly1,lz1,lelt)


      nxyz = lx1*ly1*lz1

      if (ifrhs .and. ifdp0dt) then
         call qthermal ! computes dp0thdt  
         dd = (gamma0 - 1.)/gamma0
         dd = dd * dp0thdt
         ntot = nxyz*nelfld(2)
         call invers2(dtmp,vtrans(1,1,1,1,2),ntot)
         call cmult(dtmp,dd,ntot)
         call add2 (w1,dtmp,ntot)
      endif

      j = 1
      do ifield = 2,nfield
         if (ifcvfld(ifield)) then
            ntot = nxyz*nelfld(ifield)
            call copy (y(j),w1(1,1,1,1,ifield-1),ntot)
            if (ifrhs) call col2(y(j),tmask(1,1,1,1,ifield-1),ntot)
            j = j + ntot
         endif
      enddo

      return
      end

c----------------------------------------------------------------------
      subroutine cv_upd_v
c
      include 'SIZE'
      include 'TSTEP'
      include 'DEALIAS'
      include 'SOLN'
      include 'INPUT'
      include 'CVODE'

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      ntot = lx1*ly1*lz1*nelv

      call sumab(vx,vx_,vxlag,ntot,cv_ab,nbd)
      call sumab(vy,vy_,vylag,ntot,cv_ab,nbd)
      if(if3d) call sumab(vz,vz_,vzlag,ntot,cv_ab,nbd)

      return
      end
c----------------------------------------------------------------------
      subroutine cv_upd_w
c
      include 'SIZE'
      include 'TSTEP'
      include 'MVGEOM'
      include 'INPUT'
      include 'CVODE'

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      ntot = lx1*ly1*lz1*nelv

      call sumab(wx,wx_,wxlag,ntot,cv_ab,nbd)
      call sumab(wy,wy_,wylag,ntot,cv_ab,nbd)
      if(if3d) call sumab(wz,wz_,wzlag,ntot,cv_ab,nbd)

      return
      end
c----------------------------------------------------------------------
      subroutine cv_upd_coor
c
      include 'SIZE'
      include 'TSTEP'
      include 'GEOM'
      include 'MVGEOM'
      include 'INPUT'
      include 'CVODE'

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      COMMON /SCRSF/ dtmp(lx1*ly1*lz1*lelv)

      ntot = lx1*ly1*lz1*nelv

      call sumab(dtmp,wx_,wxlag,ntot,cv_abmsh,nabmsh)
      call add3 (xm1,xm1_,dtmp,ntot)

      call sumab(dtmp,wy_,wylag,ntot,cv_abmsh,nabmsh)
      call add3 (ym1,ym1_,dtmp,ntot)

      if(if3d) then
        call sumab(dtmp,wz_,wzlag,ntot,cv_abmsh,nabmsh)
        call add3 (zm1,zm1_,dtmp,ntot)
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine my_rhs_fun (time_, y, ydot)
c
c     Compute RHS function f (called within cvode)
c     CAUTION: never touch y! 
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
      include 'CVODE'

      real time_,y(*),ydot(*)

      real w1(lx1,ly1,lz1,lelt),
     $     w2(lx1,ly1,lz1,lelt),
     $     w3(lx1,ly1,lz1,lelt)

      real ydott(lx1,ly1,lz1,lelt,ldimt)
      common /CV_YDOT/ ydott

      ifcvfun = .true.
      etime1  = dnekclock()
      time    = time_   
      nxyz    = lx1*ly1*lz1
      ntotv   = nxyz*nelv

      do ifield = 2,nfield
         ifcvfld(ifield) = .true.
      enddo
      
       
      if (time.ne.cv_timel) then
         call cv_settime     
 
         write(6,10) istep,time,time-cv_timel
  10       format(4x,i7,2x,'t=',1pE14.7,'  stepsize=',1pE13.4)

         call cv_upd_v
         call copy(w1,vx,ntotv)
         call copy(w2,vy,ntotv)
         if (if3d) call copy(w3,vz,ntotv)

         if (ifmvbd) then
            call cv_upd_coor 
            call cv_eval_geom
            call cv_upd_w
            call sub2(vx,wx,ntotv)
            call sub2(vy,wy,ntotv)
            if (if3d) call sub2(vz,wz,ntotv)
         endif
        
         if (param(99).gt.0) call set_convect_new(vxd,vyd,vzd,vx,vy,vz)

         call copy(vx,w1,ntotv)
         call copy(vy,w2,ntotv)
         if (if3d) call copy(vz,w3,ntotv)

         cv_timel = time          
      endif

      call cvunpack(t,y)          

      write(6,*) 'fcvfun',
     $                                 ifdqj

      ifield = 1
      call vprops ! we may use fluid properties somewhere
      do ifield=2,nfield
         if (ifcvfld(ifield)) call vprops
      enddo  
      do ifield=1,nfield
        write(6,*) "ifield = ", ifield
        write(6,*) "vtrans = ", vtrans(1,1,1,1,ifield)
        write(6,*) "vdiff = ", vdiff(1,1,1,1,ifield)
      enddo

      do ifield=2,nfield
         if (ifcvfld(ifield)) then
           write(6,*) "computing rhs for ifield", ifield
           ntot = nxyz*nelfld(ifield)
           call makeq
            write(6,*) "sum bq (before wt)",
     &       glsum(bq(1,1,1,1,ifield-1), ntot)

           if (iftmsh(ifield)) then                                
              call dssum(bq(1,1,1,1,ifield-1),lx1,ly1,lz1)
              call col2(bq(1,1,1,1,ifield-1),bintm1,ntot)

              call col3(w1,vtrans(1,1,1,1,ifield),bm1,ntot)      
              call dssum(w1,lx1,ly1,lz1)                        
              call col2(w1,bintm1,ntot)                           
           else                                                    
              call copy(w1,vtrans(1,1,1,1,ifield),ntot)
           endif    

           call invcol3(ydott(1,1,1,1,ifield-1),bq(1,1,1,1,ifield-1),
     &                  w1,ntot)           
         endif
      enddo

      write(6,*) "sum bq (after makeq+wt)",
     &  glsum(bq(1,1,1,1,ifield-1), ntot)

      if (ifgsh_fld_same) then ! all fields are on the v-mesh
         istride = lx1*ly1*lz1*lelt
         call nvec_dssum(ydott,istride,nfield-1,gsh_fld(1))
      else
         do ifield = 2,nfield
            if (ifcvfld(ifield) .and. gsh_fld(ifield).ge.0) then
               if(.not.iftmsh(ifield))       
     &         call dssum(ydott(1,1,1,1,ifield-1),lx1,ly1,lz1)
            endif
         enddo
      endif

      do ifield = 2,nfield
         if (ifcvfld(ifield)) then                                
           ntot = nxyz*nelfld(ifield)
           if (.not.iftmsh(ifield) .and. gsh_fld(ifield).ge.0) then
              call col2(ydott(1,1,1,1,ifield-1),binvm1,ntot)
           endif
         endif
      enddo

      call cvpack(ydot,ydott,.true.)


      tcvf = tcvf + dnekclock()-etime1 
      ncvf = ncvf + 1 

      ier = 0
      ifcvfun = .false.

      return
      end

c----------------------------------------------------------------------
      subroutine cv_eval_geom

      call glmapm1
      call geodat1
      call geom2
      call volume
      call setinvm
      call setdef

      return
      end

c----------------------------------------------------------------------
      subroutine cv_settime

      include 'SIZE'
      include 'TSTEP'
      include 'CVODE'

      cv_dtNek = time - timef ! stepsize between nek and cvode    

      cv_dtlag(1) = cv_dtNek 
      cv_dtlag(2) = dtlag(2)
      cv_dtlag(3) = dtlag(3)

      call rzero(cv_abmsh,3)
      call setabbd(cv_abmsh,cv_dtlag,nabmsh,1)
      do i = 1,3
         cv_abmsh(i) = cv_dtNek*cv_abmsh(i) 
      enddo

      call rzero(cv_ab,3)
      call setabbd(cv_ab,cv_dtlag,nbd,nbd)

      call rzero(cv_bd,4)
      call setbd(cv_bd,cv_dtlag,nbd)

      return
      end