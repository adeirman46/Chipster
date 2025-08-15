`include "fixed_point_params.vh"

//======================================================================
// Optimized CROT Gate for theta = pi/2 (Pipelined)
//======================================================================
// This module replaces the generic complex multiplier for rotations by pi/2.
// It implements multiplication by (0 + 1j), which simplifies to:
// (ar + ai*j) * (0 + 1j) = -ai + ar*j
// So, pr = -ai and pi = ar.
// This saves four multipliers and two adders compared to the generic version.
// Latency: 3 cycles (to match other gates)
module crot_pi_2_gate_pipelined(
    input                         clk,
    input                         rst_n,
    input  signed [`TOTAL_WIDTH-1:0] ar, ai,
    output signed [`TOTAL_WIDTH-1:0] pr, pi
);

    // Pipeline registers
    reg signed [`TOTAL_WIDTH-1:0] ar_s1, ai_s1;
    reg signed [`TOTAL_WIDTH-1:0] ar_s2, ai_s2;
    reg signed [`TOTAL_WIDTH-1:0] pr_s3, pi_s3;

    // Stage 1: Register inputs
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_s1 <= 0;
            ai_s1 <= 0;
        end else begin
            ar_s1 <= ar;
            ai_s1 <= ai;
        end
    end

    // Stage 2: Pass-through
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_s2 <= 0;
            ai_s2 <= 0;
        end else begin
            ar_s2 <= ar_s1;
            ai_s2 <= ai_s1;
        end
    end

    // Stage 3: Perform operation and register output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pr_s3 <= 0;
            pi_s3 <= 0;
        end else begin
            pr_s3 <= -ai_s2; // Negate
            pi_s3 <= ar_s2;  // Swap
        end
    end

    assign pr = pr_s3;
    assign pi = pi_s3;
endmodule
