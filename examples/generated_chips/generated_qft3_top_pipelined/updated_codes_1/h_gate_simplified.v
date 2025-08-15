`include "fixed_point_params.vh"

//======================================================================
// Simplified Hadamard Gate (Corrected and Pipelined)
//======================================================================
module h_gate_simplified(
    input                         clk,
    input                         rst_n,
    input  signed [`TOTAL_WIDTH-1:0] alpha_r, alpha_i,
    input  signed [`TOTAL_WIDTH-1:0] beta_r,  beta_i,
    output signed [`TOTAL_WIDTH-1:0] new_alpha_r, new_alpha_i,
    output signed [`TOTAL_WIDTH-1:0] new_beta_r,  new_beta_i
);

    // S1.4 constant for 1/sqrt(2) (0.6875)
    // 0.6875 * 2^4 = 11 (decimal).
    // In S1.4 fixed-point format (1 sign bit, 1 integer bit, 4 fractional bits),
    // 11 (decimal) is represented as `001011` (binary).
    // Interpreted as S1.4: `0*2^1 + 0*2^0 + 1*2^-1 + 0*2^-2 + 1*2^-3 + 1*2^-4 = 0.5 + 0.125 + 0.0625 = 0.6875`.
    localparam signed [`TOTAL_WIDTH-1:0] ONE_OVER_SQRT2 = 11;

    // --- Pipeline Stage 1: Addition/Subtraction ---
    reg signed [`ADD_WIDTH-1:0] add_r_s1, add_i_s1;
    reg signed [`ADD_WIDTH-1:0] sub_r_s1, sub_i_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            add_r_s1 <= 0; add_i_s1 <= 0;
            sub_r_s1 <= 0; sub_i_s1 <= 0;
        end else begin
            add_r_s1 <= alpha_r + beta_r;
            add_i_s1 <= alpha_i + beta_i;
            sub_r_s1 <= alpha_r - beta_r;
            sub_i_s1 <= alpha_i - beta_i;
        end
    end

    // --- Pipeline Stage 2: Multiplication by 1/sqrt(2) using operator strength reduction ---
    reg signed [`MULT_RESULT_WIDTH-1:0] mult_add_r_s2, mult_add_i_s2;
    reg signed [`MULT_RESULT_WIDTH-1:0] mult_sub_r_s2, mult_sub_i_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_add_r_s2 <= 0; mult_add_i_s2 <= 0;
            mult_sub_r_s2 <= 0; mult_sub_i_s2 <= 0;
        end else begin
            // Replacement of X * 11 with (X << 3) + (X << 1) + X
            // For signed numbers, standard Verilog `<<` performs arithmetic left shift
            // as long as the result fits the target width. The sum result is assigned
            // to a wider register (`MULT_RESULT_WIDTH`) to prevent overflow.
            mult_add_r_s2 <= (add_r_s1 << 3) + (add_r_s1 << 1) + add_r_s1;
            mult_add_i_s2 <= (add_i_s1 << 3) + (add_i_s1 << 1) + add_i_s1;
            mult_sub_r_s2 <= (sub_r_s1 << 3) + (sub_r_s1 << 1) + sub_r_s1;
            mult_sub_i_s2 <= (sub_i_s1 << 3) + (sub_i_s1 << 1) + sub_i_s1;
        end
    end

    // --- Pipeline Stage 3: Scaling (Output) ---
    reg signed [`TOTAL_WIDTH-1:0] new_alpha_r_s3, new_alpha_i_s3;
    reg signed [`TOTAL_WIDTH-1:0] new_beta_r_s3,  new_beta_i_s3;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            new_alpha_r_s3 <= 0; new_alpha_i_s3 <= 0;
            new_beta_r_s3  <= 0; new_beta_i_s3  <= 0;
        end else begin
            // Arithmetic right shift by FRAC_WIDTH bits to scale back to original fixed-point format.
            // Implicit truncation to `TOTAL_WIDTH` bits.
            new_alpha_r_s3 <= mult_add_r_s2 >>> `FRAC_WIDTH;
            new_alpha_i_s3 <= mult_add_i_s2 >>> `FRAC_WIDTH;
            new_beta_r_s3  <= mult_sub_r_s2 >>> `FRAC_WIDTH;
            new_beta_i_s3  <= mult_sub_i_s2 >>> `FRAC_WIDTH;
        end
    end
    
    assign new_alpha_r = new_alpha_r_s3;
    assign new_alpha_i = new_alpha_i_s3;
    assign new_beta_r  = new_beta_r_s3;
    assign new_beta_i  = new_beta_i_s3;
    
endmodule
