`include "fixed_point_params.vh"
//======================================================================
// SWAP Gate (Pipelined)
//======================================================================
module swap_gate_pipelined(
    input                         clk,
    input                         rst_n,
    input  signed [`TOTAL_WIDTH-1:0] in_001_r, in_001_i,
    input  signed [`TOTAL_WIDTH-1:0] in_100_r, in_100_i,
    input  signed [`TOTAL_WIDTH-1:0] in_011_r, in_011_i,
    input  signed [`TOTAL_WIDTH-1:0] in_110_r, in_110_i,
    output signed [`TOTAL_WIDTH-1:0] out_001_r, out_001_i,
    output signed [`TOTAL_WIDTH-1:0] out_100_r, out_100_i,
    output signed [`TOTAL_WIDTH-1:0] out_011_r, out_011_i,
    output signed [`TOTAL_WIDTH-1:0] out_110_r, out_110_i
);

    reg signed [`TOTAL_WIDTH-1:0] out_001_r_reg, out_001_i_reg;
    reg signed [`TOTAL_WIDTH-1:0] out_100_r_reg, out_100_i_reg;
    reg signed [`TOTAL_WIDTH-1:0] out_011_r_reg, out_011_i_reg;
    reg signed [`TOTAL_WIDTH-1:0] out_110_r_reg, out_110_i_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_001_r_reg <= 0; out_001_i_reg <= 0;
            out_100_r_reg <= 0; out_100_i_reg <= 0;
            out_011_r_reg <= 0; out_011_i_reg <= 0;
            out_110_r_reg <= 0; out_110_i_reg <= 0;
        end else begin
            // Perform swaps
            out_001_r_reg <= in_100_r; out_001_i_reg <= in_100_i;
            out_100_r_reg <= in_001_r; out_100_i_reg <= in_001_i;
            out_011_r_reg <= in_110_r; out_011_i_reg <= in_110_i;
            out_110_r_reg <= in_011_r; out_110_i_reg <= in_011_i;
        end
    end

    assign out_001_r = out_001_r_reg;
    assign out_001_i = out_001_i_reg;
    assign out_100_r = out_100_r_reg;
    assign out_100_i = out_100_i_reg;
    assign out_011_r = out_011_r_reg;
    assign out_011_i = out_011_i_reg;
    assign out_110_r = out_110_r_reg;
    assign out_110_i = out_110_i_reg;

endmodule
