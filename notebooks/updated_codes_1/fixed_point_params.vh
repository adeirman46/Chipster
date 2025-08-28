//======================================================================
// PARAMETER DEFINITIONS
//======================================================================

// Total bits for our signed fixed-point number
`define TOTAL_WIDTH 6
// Number of fractional bits
`define FRAC_WIDTH 4
// Width for intermediate addition results
`define ADD_WIDTH (`TOTAL_WIDTH + 1)
// Width for intermediate multiplication results after operator strength reduction
// For X * 11 = (X << 3) + (X << 1) + X
// X is ADD_WIDTH bits. X << 3 is ADD_WIDTH + 3 bits.
// Sum needs ADD_WIDTH + 3 + 1 = ADD_WIDTH + 4 bits.
`define MULT_RESULT_WIDTH (`ADD_WIDTH + 4)
