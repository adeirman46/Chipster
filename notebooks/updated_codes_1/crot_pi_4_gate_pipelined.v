`include "fixed_point_params.vh"
//======================================================================
// Optimized CROT Gate for theta = pi/4 (Pipelined)
//======================================================================
module crot_pi_4_gate_pipelined(
    input                         clk,
    input                         rst_n,
    input  signed [`TOTAL_WIDTH-1:0] ar, ai,
    output signed [`TOTAL_WIDTH-1:0] pr, pi
);
    // Constant for ~0.707 (1/sqrt(2)) in S1.4 format (0.6875)
    localparam signed [`TOTAL_WIDTH-1:0] C_VAL = 11;

    // Intermediate widths
    localparam CROT_ADD_WIDTH = `ADD_WIDTH;
    localparam CROT_MULT_WIDTH = `MULT_RESULT_WIDTH;

    // Stage 1: Addition/Subtraction
    reg signed [CROT_ADD_WIDTH-1:0] sub_s1, add_s1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sub_s1 <= 0;
            add_s1 <= 0;
        end else begin
            sub_s1 <= ar - ai;
            add_s1 <= ar + ai;
        end
    end

    // Stage 2: Multiplication by constant using operator strength reduction
    reg signed [CROT_MULT_WIDTH-1:0] pr_prod_s2, pi_prod_s2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pr_prod_s2 <= 0;
            pi_prod_s2 <= 0;
        end else begin
            pr_prod_s2 <= (sub_s1 << 3) + (sub_s1 << 1) + sub_s1;
            pi_prod_s2 <= (add_s1 << 3) + (add_s1 << 1) + add_s1;
        end
    end
    
    // Stage 3: Scaling and Output
    reg signed [`TOTAL_WIDTH-1:0] pr_s3, pi_s3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pr_s3 <= 0;
            pi_s3 <= 0;
        end else begin
            pr_s3 <= pr_prod_s2 >>> `FRAC_WIDTH;
            pi_s3 <= pi_prod_s2 >>> `FRAC_WIDTH;
        end
    end
    
    assign pr = pr_s3;
    assign pi = pi_s3;
endmodule
