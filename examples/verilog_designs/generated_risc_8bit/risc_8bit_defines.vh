// Shared parameters and opcodes for the 8-bit RISC processor

//================================================================
// Parameters
//================================================================
`define D_WIDTH        8
`define A_WIDTH        8
`define REG_ADDR_WIDTH 4
`define REG_COUNT      (1 << `REG_ADDR_WIDTH)
`define IMEM_SIZE      256
`define DMEM_SIZE      256

// Instruction Opcodes
`define OP_NOP  4'h0 // No Operation
`define OP_ADD  4'h1 // Add
`define OP_SUB  4'h2 // Subtract
`define OP_AND  4'h3 // Bitwise AND
`define OP_OR   4'h4 // Bitwise OR
`define OP_SL   4'h5 // Shift Left by 1
`define OP_LW   4'h6 // Load Word
`define OP_SW   4'h7 // Store Word
`define OP_ADDI 4'h8 // Add Immediate
`define OP_JMP  4'h9 // Jump (not implemented in pipeline)
`define OP_BEQ  4'hA // Branch if Equal
`define OP_HALT 4'hF // Halt Processor