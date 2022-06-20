Tasks:
1. validate rhs implementation in unit test
    - include source term, w/ moving mesh, ...
2. add in cvode interface/calls


Pending TODOs:

- Apply RHS operators to multiple fields at once
  * Poor man's solution: 2 outer loops over Nelements, Nfields

- Correctly lump CVODE scalars to minimize number of gather-scatter calls

- Determine how to perform overlap with userLocalPointSource without
  introducing significant memory overhead