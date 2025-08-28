/*
 * Verilog code for a simple 8-bit, 5-stage pipelined RISC processor.
 *
 * This monolithic module defines a complete processor including:
 * - A simple 16-bit instruction set architecture (ISA).
 * - 5-stage pipeline: Fetch, Decode, Execute, Memory, Write-back.
 * - Internal instruction and data memories, pre-loaded with a test program.
 * - 16 general-purpose 8-bit registers.
 * - Hazard detection unit for load-use stalls.
 * - Forwarding unit to mitigate data hazards.
 * - Branch prediction (branch not taken) with pipeline flushing on taken branches.
 */
module risc_8bit_processor (
    input wire clk,
    input wire rst
);

    // --- Architectural Parameters (previously in risc_8bit_defines.vh) ---
    parameter D_WIDTH = 8;         // Data Width
    parameter A_WIDTH = 8;         // Address Width (for PC and Data Memory)
    parameter REG_COUNT = 16;      // Number of Registers
    parameter REG_ADDR_WIDTH = 4;  // Bits to address a register
    parameter IMEM_SIZE = 256;     // Instruction Memory Size
    parameter DMEM_SIZE = 256;     // Data Memory Size

    // --- Instruction Opcodes (previously in risc_8bit_defines.vh) ---
    localparam OP_NOP  = 4'h0;
    localparam OP_ADDI = 4'h1;
    localparam OP_ADD  = 4'h2;
    localparam OP_SW   = 4'h3;
    localparam OP_LW   = 4'h4;
    localparam OP_SUB  = 4'h5;
    localparam OP_BEQ  = 4'h6;
    localparam OP_AND  = 4'h7;
    localparam OP_OR   = 4'h8;
    localparam OP_SL   = 4'h9;
    localparam OP_HALT = 4'hF;

    //================================================================
    // Memories and Register File
    //================================================================
    reg [A_WIDTH-1:0] pc;
    reg [15:0]        instr_mem [0:IMEM_SIZE-1];
    reg [D_WIDTH-1:0] data_mem [0:DMEM_SIZE-1];
    reg [D_WIDTH-1:0] reg_file [0:REG_COUNT-1];
    reg               halted;

    //================================================================
    // Pipeline Wires and Registers
    //================================================================

    // --- IF Stage ---
    wire [15:0]       if_instr;
    wire [A_WIDTH-1:0] pc_next;
    wire [A_WIDTH-1:0] pc_plus_1;

    // --- IF/ID Pipeline Register ---
    reg [15:0]        if_id_instr;
    reg [A_WIDTH-1:0] if_id_pc_plus_1;
    reg               if_id_valid;

    // --- ID Stage ---
    wire [3:0]                 id_opcode;
    wire [REG_ADDR_WIDTH-1:0]  id_rs1_addr, id_rs2_addr, id_rd_addr;
    wire [D_WIDTH-1:0]         id_rs1_data, id_rs2_data;
    wire [3:0]                 id_imm;
    // Control Signals
    wire                       id_reg_write;
    wire [2:0]                 id_alu_op;
    wire                       id_alu_src; // 0 for reg, 1 for imm
    wire                       id_mem_read;
    wire                       id_mem_write;
    wire                       id_mem_to_reg;
    wire                       id_branch;

    // --- ID/EX Pipeline Register ---
    reg                        id_ex_reg_write;
    reg [2:0]                  id_ex_alu_op;
    reg                        id_ex_alu_src;
    reg                        id_ex_mem_read;
    reg                        id_ex_mem_write;
    reg                        id_ex_mem_to_reg;
    reg                        id_ex_branch;
    reg [A_WIDTH-1:0]          id_ex_pc_plus_1;
    reg [D_WIDTH-1:0]          id_ex_rs1_data;
    reg [D_WIDTH-1:0]          id_ex_rs2_data;
    reg [3:0]                  id_ex_imm;
    reg [REG_ADDR_WIDTH-1:0]   id_ex_rs1_addr, id_ex_rs2_addr, id_ex_rd_addr;

    // --- EX Stage ---
    wire [D_WIDTH-1:0]         ex_alu_in1, ex_alu_in2;
    wire [D_WIDTH-1:0]         ex_alu_result;
    wire                       ex_zero_flag;
    wire [A_WIDTH-1:0]         ex_branch_addr;
    wire                       ex_branch_taken;
    wire [1:0]                 forwardA, forwardB;

    // --- EX/MEM Pipeline Register ---
    reg                        ex_mem_reg_write;
    reg                        ex_mem_mem_read;
    reg                        ex_mem_mem_write;
    reg                        ex_mem_mem_to_reg;
    reg [D_WIDTH-1:0]          ex_mem_alu_result;
    reg [D_WIDTH-1:0]          ex_mem_store_data; // Data for SW
    reg [REG_ADDR_WIDTH-1:0]   ex_mem_rd_addr;
    reg                        ex_mem_branch_taken;

    // --- MEM Stage ---
    wire [D_WIDTH-1:0]         mem_read_data;

    // --- MEM/WB Pipeline Register ---
    reg                        mem_wb_reg_write;
    reg                        mem_wb_mem_to_reg;
    reg [D_WIDTH-1:0]          mem_wb_read_data;
    reg [D_WIDTH-1:0]          mem_wb_alu_result;
    reg [REG_ADDR_WIDTH-1:0]   mem_wb_rd_addr;

    // --- WB Stage ---
    wire [D_WIDTH-1:0]         wb_write_data;

    // --- Hazard Unit Signals ---
    wire                       pc_stall;
    wire                       id_ex_bubble;
    wire                       if_id_flush;

    //================================================================
    // Memory and Register Initialization
    //================================================================
    integer i;
    initial begin
        // Load a simple test program into instruction memory
        // Program: R1=5, R2=10, R3=R1+R2, store R3 at mem[20], load it back to R5, branch, halt.
        // Instruction Format: {opcode[15:12], rd[11:8], rs1[7:4], rs2/imm[3:0]}
        for (i = 0; i < IMEM_SIZE; i = i + 1) instr_mem[i] = {OP_NOP, 12'h000};
        
        instr_mem[0] = {OP_ADDI, 4'd1, 4'd0, 4'd5};    // 0: ADDI R1, R0, 5   (R0 is always 0)
        instr_mem[1] = {OP_ADDI, 4'd2, 4'd0, 4'd10};   // 1: ADDI R2, R0, 10
        instr_mem[2] = {OP_ADD,  4'd3, 4'd1, 4'd2};    // 2: ADD R3, R1, R2
        instr_mem[3] = {OP_ADDI, 4'd4, 4'd0, 4'd12};   // 3: ADDI R4, R0, 12 (address for store, NOTE: immediate must be 4-bit)
        instr_mem[4] = {OP_SW,   4'd3, 4'd4, 4'd0};    // 4: SW R3, 0(R4)  (rs2=R3, rs1=R4)
        instr_mem[5] = {OP_LW,   4'd5, 4'd4, 4'd0};    // 5: LW R5, 0(R4)
        instr_mem[6] = {OP_SUB,  4'd6, 4'd2, 4'd1};    // 6: SUB R6, R2, R1  (R6 should be 5)
        instr_mem[7] = {OP_BEQ,  4'd1, 4'd6, 4'd2};    // 7: BEQ R1, R6, 2   (branch to instr 10)
        instr_mem[8] = {OP_NOP,  12'h000};             // 8: NOP (should be skipped)
        instr_mem[9] = {OP_NOP,  12'h000};             // 9: NOP (should be skipped)
        instr_mem[10]= {OP_HALT, 12'hFFF};             // 10: HALT

        // Initialize data memory and registers to 0
        for (i = 0; i < DMEM_SIZE; i = i + 1) data_mem[i] = 8'h00;
        for (i = 0; i < REG_COUNT; i = i + 1) reg_file[i] = 8'h00;
    end

    //================================================================
    // STAGE 1: INSTRUCTION FETCH (IF)
    //================================================================
    assign pc_plus_1 = pc + 1;
    assign if_instr = instr_mem[pc];

    assign pc_next = ex_branch_taken ? ex_branch_addr : pc_plus_1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pc <= 0;
            halted <= 1'b0;
        end else if (~pc_stall && ~halted) begin
            pc <= pc_next;
        end
        
        // Halt logic is latched when HALT instruction is in ID stage
        if (~rst && if_id_valid && (id_opcode == OP_HALT)) begin
            halted <= 1'b1;
        end
    end

    //================================================================
    // IF/ID PIPELINE REGISTER
    //================================================================
    always @(posedge clk or posedge rst) begin
        if (rst || if_id_flush) begin
            if_id_instr <= {OP_NOP, 12'h0};
            if_id_pc_plus_1 <= 0;
            if_id_valid <= 1'b0;
        end else if (~pc_stall) begin
            if_id_instr <= if_instr;
            if_id_pc_plus_1 <= pc_plus_1;
            if_id_valid <= ~halted;
        end
    end

    //================================================================
    // STAGE 2: INSTRUCTION DECODE (ID)
    //================================================================
    assign id_opcode   = if_id_instr[15:12];
    assign id_rd_addr  = if_id_instr[11:8];
    assign id_rs1_addr = if_id_instr[7:4];
    assign id_rs2_addr = if_id_instr[3:0];
    assign id_imm      = if_id_instr[3:0];

    // Register File Read Ports (R0 is always 0)
    assign id_rs1_data = (id_rs1_addr == 0) ? 0 : reg_file[id_rs1_addr];
    assign id_rs2_data = (id_rs2_addr == 0) ? 0 : reg_file[id_rs2_addr];

    // Control Unit Logic (combinational)
    assign id_reg_write = (id_opcode == OP_ADD || id_opcode == OP_SUB || id_opcode == OP_AND || id_opcode == OP_OR || id_opcode == OP_SL || id_opcode == OP_LW || id_opcode == OP_ADDI) && if_id_valid;
    assign id_alu_src   = (id_opcode == OP_ADDI || id_opcode == OP_LW || id_opcode == OP_SW);
    assign id_mem_to_reg= (id_opcode == OP_LW);
    assign id_mem_read  = (id_opcode == OP_LW);
    assign id_mem_write = (id_opcode == OP_SW);
    assign id_branch    = (id_opcode == OP_BEQ);

    // ALU Control
    assign id_alu_op = (id_opcode == OP_ADD || id_opcode == OP_ADDI || id_opcode == OP_LW || id_opcode == OP_SW) ? 3'b001 : // ADD
                       (id_opcode == OP_SUB || id_opcode == OP_BEQ) ? 3'b010 : // SUB
                       (id_opcode == OP_AND) ? 3'b011 : // AND
                       (id_opcode == OP_OR)  ? 3'b100 : // OR
                       (id_opcode == OP_SL)  ? 3'b101 : // SL
                       3'b000; // Default

    //================================================================
    // ID/EX PIPELINE REGISTER
    //================================================================
    always @(posedge clk or posedge rst) begin
        if (rst || id_ex_bubble) begin
            id_ex_reg_write <= 0;
            id_ex_alu_op <= 3'b000;
            id_ex_alu_src <= 0;
            id_ex_mem_read <= 0;
            id_ex_mem_write <= 0;
            id_ex_mem_to_reg <= 0;
            id_ex_branch <= 0;
            // Data path registers are not cleared to save logic, as control signals are 0
            id_ex_rd_addr <= 0; // Prevent forwarding from invalid instruction
        end else begin
            id_ex_reg_write <= id_reg_write;
            id_ex_alu_op <= id_alu_op;
            id_ex_alu_src <= id_alu_src;
            id_ex_mem_read <= id_mem_read;
            id_ex_mem_write <= id_mem_write;
            id_ex_mem_to_reg <= id_mem_to_reg;
            id_ex_branch <= id_branch;
            id_ex_pc_plus_1 <= if_id_pc_plus_1;
            id_ex_rs1_data <= id_rs1_data;
            id_ex_rs2_data <= id_rs2_data;
            id_ex_imm <= id_imm;
            id_ex_rs1_addr <= id_rs1_addr;
            id_ex_rs2_addr <= id_rs2_addr;
            id_ex_rd_addr <= id_rd_addr;
        end
    end

    //================================================================
    // STAGE 3: EXECUTE (EX)
    //================================================================

    // Forwarding MUXes for ALU inputs
    assign ex_alu_in1 = (forwardA == 2'b01) ? ex_mem_alu_result :
                        (forwardA == 2'b10) ? wb_write_data :
                        id_ex_rs1_data; // Default: 2'b00

    wire [D_WIDTH-1:0] ex_alu_in2_base = id_ex_alu_src ? {{D_WIDTH-4{id_ex_imm[3]}}, id_ex_imm} : id_ex_rs2_data; // Sign-extend immediate

    assign ex_alu_in2 = (forwardB == 2'b01) ? ex_mem_alu_result :
                        (forwardB == 2'b10) ? wb_write_data :
                        ex_alu_in2_base; // Default: 2'b00

    // ALU
    reg [D_WIDTH-1:0] ex_alu_result_comb;
    always @(*) begin
        case (id_ex_alu_op)
            3'b001: ex_alu_result_comb = ex_alu_in1 + ex_alu_in2; // ADD
            3'b010: ex_alu_result_comb = ex_alu_in1 - ex_alu_in2; // SUB
            3'b011: ex_alu_result_comb = ex_alu_in1 & ex_alu_in2; // AND
            3'b100: ex_alu_result_comb = ex_alu_in1 | ex_alu_in2; // OR
            3'b101: ex_alu_result_comb = ex_alu_in1 << 1;         // SL
            default: ex_alu_result_comb = 8'h00;
        endcase
    end
    assign ex_alu_result = ex_alu_result_comb;

    assign ex_zero_flag = (ex_alu_in1 == ex_alu_in2); // For BEQ

    // Branch logic
    assign ex_branch_addr = id_ex_pc_plus_1 + {{A_WIDTH-4{id_ex_imm[3]}}, id_ex_imm};
    assign ex_branch_taken = id_ex_branch & ex_zero_flag;

    //================================================================
    // EX/MEM PIPELINE REGISTER
    //================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ex_mem_reg_write <= 0;
            ex_mem_mem_read <= 0;
            ex_mem_mem_write <= 0;
            ex_mem_mem_to_reg <= 0;
            ex_mem_alu_result <= 0;
            ex_mem_store_data <= 0;
            ex_mem_rd_addr <= 0;
            ex_mem_branch_taken <= 0;
        end else begin
            ex_mem_reg_write <= id_ex_reg_write;
            ex_mem_mem_read <= id_ex_mem_read;
            ex_mem_mem_write <= id_ex_mem_write;
            ex_mem_mem_to_reg <= id_ex_mem_to_reg;
            ex_mem_alu_result <= ex_alu_result;
            ex_mem_store_data <= ex_alu_in2; // Forwarded value for SW
            ex_mem_rd_addr <= id_ex_rd_addr;
            ex_mem_branch_taken <= ex_branch_taken;
        end
    end

    //================================================================
    // STAGE 4: MEMORY ACCESS (MEM)
    //================================================================
    assign mem_read_data = data_mem[ex_mem_alu_result];

    // Data Memory Write Port
    always @(posedge clk) begin
        if (~rst && ex_mem_mem_write) begin
            data_mem[ex_mem_alu_result] <= ex_mem_store_data;
        end
    end

    //================================================================
    // MEM/WB PIPELINE REGISTER
    //================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mem_wb_reg_write <= 0;
            mem_wb_mem_to_reg <= 0;
            mem_wb_read_data <= 0;
            mem_wb_alu_result <= 0;
            mem_wb_rd_addr <= 0;
        end else begin
            mem_wb_reg_write <= ex_mem_reg_write;
            mem_wb_mem_to_reg <= ex_mem_mem_to_reg;
            mem_wb_read_data <= mem_read_data;
            mem_wb_alu_result <= ex_mem_alu_result;
            mem_wb_rd_addr <= ex_mem_rd_addr;
        end
    end

    //================================================================
    // STAGE 5: WRITE BACK (WB)
    //================================================================
    assign wb_write_data = mem_wb_mem_to_reg ? mem_wb_read_data : mem_wb_alu_result;

    // Register File Write Port
    always @(posedge clk) begin
        if (~rst && mem_wb_reg_write && (mem_wb_rd_addr != 0)) begin
            reg_file[mem_wb_rd_addr] <= wb_write_data;
        end
    end

    //================================================================
    // HAZARD DETECTION AND FORWARDING UNIT
    //================================================================

    // Stall logic for load-use hazard
    wire load_use_hazard = (id_ex_mem_read &&
                           ((id_ex_rd_addr == id_rs1_addr && id_rs1_addr != 0) ||
                            (id_ex_rd_addr == id_rs2_addr && id_rs2_addr != 0 && !id_alu_src)));

    assign pc_stall = load_use_hazard;
    assign id_ex_bubble = load_use_hazard || ex_branch_taken;
    assign if_id_flush = ex_branch_taken;

    // Forwarding logic for EX stage
    // Forward from EX/MEM stage (priority 1)
    assign forwardA = (ex_mem_reg_write && (ex_mem_rd_addr != 0) && (ex_mem_rd_addr == id_ex_rs1_addr)) ? 2'b01 :
                      // Forward from MEM/WB stage (priority 2)
                      (mem_wb_reg_write && (mem_wb_rd_addr != 0) && (mem_wb_rd_addr == id_ex_rs1_addr) && !(ex_mem_reg_write && (ex_mem_rd_addr != 0) && (ex_mem_rd_addr == id_ex_rs1_addr))) ? 2'b10 :
                      2'b00; // No forwarding

    assign forwardB = (ex_mem_reg_write && (ex_mem_rd_addr != 0) && (ex_mem_rd_addr == id_ex_rs2_addr) && !id_ex_alu_src) ? 2'b01 :
                      // Forward from MEM/WB stage
                      (mem_wb_reg_write && (mem_wb_rd_addr != 0) && (mem_wb_rd_addr == id_ex_rs2_addr) && !id_ex_alu_src && !(ex_mem_reg_write && (ex_mem_rd_addr != 0) && (ex_mem_rd_addr == id_ex_rs2_addr))) ? 2'b10 :
                      2'b00; // No forwarding

endmodule