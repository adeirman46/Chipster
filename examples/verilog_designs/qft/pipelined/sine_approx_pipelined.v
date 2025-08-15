`include "fixed_point_params.vh"

module sine_approx_pipelined(
    input                         clk,
    input                         rst_n,
    input  wire signed [`TOTAL_WIDTH-1:0] x,
    output wire signed [`TOTAL_WIDTH-1:0] y
);

    // Breakpoints and constants
    localparam signed [`TOTAL_WIDTH-1:0]
        BP_N101 = -101, BP_N88 = -88, BP_N75 = -75, BP_N63 = -63,
        BP_N50 = -50,  BP_N38 = -38, BP_N25 = -25, BP_N13 = -13,
        BP_0   = 0,    BP_P13 = 13,  BP_P25 = 25,  BP_P38 = 38,
        BP_P50 = 50,  BP_P63 = 63,  BP_P75 = 75,  BP_P88 = 88;
    localparam signed [`TOTAL_WIDTH-1:0]
        S0 = 14, S1 = 6,  S2 = -6, S3 = -14, S4 = -14, S5 = -6,
        S6 = 6,  S7 = 14, S8 = 14, S9 = 6,   S10= -6,  S11= -14,
        S12= -14,S13= -6, S14= 6,  S15= 14;
    localparam signed [`TOTAL_WIDTH-1:0]
        I0 = 91, I1 = 44,  I2 = -12, I3 = -45, I4 = -45, I5 = -25,
        I6 = -7, I7 = 0,   I8 = 0,   I9 = 7,   I10= 25,  I11= 45,
        I12= 45, I13= 12,  I14= -44, I15= -91;

    // --- Pipeline Stage 1: Register input, select slope/intercept ---
    reg signed [`TOTAL_WIDTH-1:0] x_s1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) x_s1 <= 0;
        else        x_s1 <= x;
    end

    wire signed [`TOTAL_WIDTH-1:0] slope_s1, intercept_s1;
    assign slope_s1 = (x_s1 < BP_N88) ? S0 : (x_s1 < BP_N75) ? S1 : (x_s1 < BP_N63) ? S2 : (x_s1 < BP_N50) ? S3 :
                      (x_s1 < BP_N38) ? S4 : (x_s1 < BP_N25) ? S5 : (x_s1 < BP_N13) ? S6 : (x_s1 < BP_0)   ? S7 :
                      (x_s1 < BP_P13) ? S8 : (x_s1 < BP_P25) ? S9 : (x_s1 < BP_P38) ? S10: (x_s1 < BP_P50) ? S11:
                      (x_s1 < BP_P63) ? S12: (x_s1 < BP_P75) ? S13: (x_s1 < BP_P88) ? S14: S15;
    assign intercept_s1 = (x_s1 < BP_N88) ? I0 : (x_s1 < BP_N75) ? I1 : (x_s1 < BP_N63) ? I2 : (x_s1 < BP_N50) ? I3 :
                          (x_s1 < BP_N38) ? I4 : (x_s1 < BP_N25) ? I5 : (x_s1 < BP_N13) ? I6 : (x_s1 < BP_0)   ? I7 :
                          (x_s1 < BP_P13) ? I8 : (x_s1 < BP_P25) ? I9 : (x_s1 < BP_P38) ? I10: (x_s1 < BP_P50) ? I11:
                          (x_s1 < BP_P63) ? I12: (x_s1 < BP_P75) ? I13: (x_s1 < BP_P88) ? I14: I15;

    // --- Pipeline Stage 2: Multiplication ---
    reg signed [`MULT_WIDTH-1:0] temp_mult_s2;
    reg signed [`TOTAL_WIDTH-1:0] intercept_s2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temp_mult_s2 <= 0;
            intercept_s2 <= 0;
        end else begin
            temp_mult_s2 <= x_s1 * slope_s1;
            intercept_s2 <= intercept_s1;
        end
    end

    // --- Pipeline Stage 3: Scaling and Addition (Output) ---
    reg signed [`TOTAL_WIDTH-1:0] y_s3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_s3 <= 0;
        end else begin
            y_s3 <= (temp_mult_s2 >>> `FRAC_WIDTH) + intercept_s2;
        end
    end

    assign y = y_s3;

endmodule