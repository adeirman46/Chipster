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

    // S4.4 constant for 1/sqrt(2)
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

    // --- Pipeline Stage 2: Multiplication by 1/sqrt(2) ---
    localparam H_MULT_WIDTH = `ADD_WIDTH + `TOTAL_WIDTH;
    reg signed [H_MULT_WIDTH-1:0] mult_add_r_s2, mult_add_i_s2;
    reg signed [H_MULT_WIDTH-1:0] mult_sub_r_s2, mult_sub_i_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_add_r_s2 <= 0; mult_add_i_s2 <= 0;
            mult_sub_r_s2 <= 0; mult_sub_i_s2 <= 0;
        end else begin
            mult_add_r_s2 <= add_r_s1 * ONE_OVER_SQRT2;
            mult_add_i_s2 <= add_i_s1 * ONE_OVER_SQRT2;
            mult_sub_r_s2 <= sub_r_s1 * ONE_OVER_SQRT2;
            mult_sub_i_s2 <= sub_i_s1 * ONE_OVER_SQRT2;
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
