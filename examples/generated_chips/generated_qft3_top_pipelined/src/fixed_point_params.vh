//
// fixed_point_params.vh
// Defines the fixed-point data type for the QFT project.
//

// Total bits for our signed fixed-point number
`define TOTAL_WIDTH 8

// Number of fractional bits
`define FRAC_WIDTH 4

// Width for intermediate multiplication results (before scaling)
`define MULT_WIDTH (`TOTAL_WIDTH * 2)

// Width for intermediate addition results
`define ADD_WIDTH (`TOTAL_WIDTH + 1)
