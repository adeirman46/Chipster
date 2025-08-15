`include "fixed_point_params.vh"

module crot_gate_pipelined(
    input                         clk,
    input                         rst_n,
    input  wire signed [`TOTAL_WIDTH-1:0] in_r, in_i,
    input  wire signed [`TOTAL_WIDTH-1:0] theta,
    output wire signed [`TOTAL_WIDTH-1:0] out_r, out_i
);

    // Latency of the trig approximation modules is 3 stages
    localparam TRIG_LATENCY = 3;

    // --- Step 1: Calculate cos(theta) and sin(theta) ---
    wire signed [`TOTAL_WIDTH-1:0] cos_theta;
    wire signed [`TOTAL_WIDTH-1:0] sin_theta;

    cosine_approx_pipelined cos_unit (
        .clk(clk), .rst_n(rst_n), .x(theta), .y(cos_theta)
    );
    sine_approx_pipelined sin_unit (
        .clk(clk), .rst_n(rst_n), .x(theta), .y(sin_theta)
    );

    // --- Delay line for in_r and in_i to match TRIG_LATENCY ---
    reg signed [`TOTAL_WIDTH-1:0] in_r_delay [TRIG_LATENCY-1:0];
    reg signed [`TOTAL_WIDTH-1:0] in_i_delay [TRIG_LATENCY-1:0];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<TRIG_LATENCY; i=i+1) begin
                in_r_delay[i] <= 0;
                in_i_delay[i] <= 0;
            end
        end else begin
            in_r_delay[0] <= in_r;
            in_i_delay[0] <= in_i;
            for (i=1; i<TRIG_LATENCY; i=i+1) begin
                in_r_delay[i] <= in_r_delay[i-1];
                in_i_delay[i] <= in_i_delay[i-1];
            end
        end
    end

    // --- Step 2: Perform the complex multiplication ---
    ccmult_pipelined rotation_multiplier (
        .clk(clk), .rst_n(rst_n),
        .ar(in_r_delay[TRIG_LATENCY-1]), .ai(in_i_delay[TRIG_LATENCY-1]),
        .br(cos_theta),                  .bi(sin_theta),
        .pr(out_r),                      .pi(out_i)
    );

endmodule