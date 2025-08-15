`include "fixed_point_params.vh"

module cadd_pipelined(
    input                         clk,    // Clock
    input                         rst_n,  // Asynchronous reset, active-low
    input  signed [`TOTAL_WIDTH-1:0] ar, ai,   // Input A
    input  signed [`TOTAL_WIDTH-1:0] br, bi,   // Input B
    output signed [`ADD_WIDTH-1:0]   pr, pi    // Pipelined Output P
);

    reg signed [`ADD_WIDTH-1:0] pr_reg, pi_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pr_reg <= 0;
            pi_reg <= 0;
        end else begin
            pr_reg <= ar + br;
            pi_reg <= ai + bi;
        end
    end

    assign pr = pr_reg;
    assign pi = pi_reg;

endmodule