      COMMON /SCNRS/ sc_nrs(20)
      real               sc_nrs
      real NLAT
      real UREF
      real LREF
      real TREF
      real TS0
      real SCR
      real ILSTRENGTH
      integer NUMBER_ELEMENTS_X
      integer NUMBER_ELEMENTS_Y
      integer NUMBER_ELEMENTS_Z
      real XLEN
      real YLEN
      real ZLEN
      real BETAM
      NLAT              =     sc_nrs(1)
      UREF              =     sc_nrs(2)
      LREF              =     sc_nrs(3)
      TREF              =     sc_nrs(4)
      TS0               =     sc_nrs(5)
      SCR               =     sc_nrs(6)
      ILSTRENGTH        =     sc_nrs(7)
      NUMBER_ELEMENTS_X = int(sc_nrs(8))
      NUMBER_ELEMENTS_Y = int(sc_nrs(9))
      NUMBER_ELEMENTS_Z = int(sc_nrs(10))
      XLEN              =     sc_nrs(11)
      YLEN              =     sc_nrs(12)
      ZLEN              =     sc_nrs(13)
      BETAM             =     sc_nrs(14)
