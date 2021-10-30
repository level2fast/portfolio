1. Purpose

This project is intended to help you understand in detail how a pipelined
implementation works.  You will write a cycle-accurate simulator for a
pipelined implementation of the LC that performs dynamic instruction
scheduling.

2. LC Instruction-Set Architecture

For the CSCI 560 programming assignments, you will be using the LC
(Little Computer). The LC is very simple, but it is general
enough to solve complex problems. For this project, you will only need to know
the instruction set and instruction format of the LC.

The LC is an 8-register, 32-bit computer.  All addresses are
word-addresses.  The LC has 65536 words of memory.  By assembly-language
convention, register 0 will always contain the value 0.

There are 4 instruction formats (bit 0 is the least-significant bit).  Bits
31-25 are unused for all instructions, and should always be 0.

R-type instructions (add, nand, mult):
    bits 24-22: opcode
    bits 21-19: reg A
    bits 18-16: reg B
    bits 15-3:  unused (should all be 0)
    bits 2-0:   destReg

I-type instructions (lw, sw, beq):
    bits 24-22: opcode
    bits 21-19: reg A
    bits 18-16: reg B
    bits 15-0:  offsetField (an 16-bit, 2's complement number with a range of
		    -32768 to 32767)

O-type instructions (halt, noop):
    bits 24-22: opcode
    bits 21-0:  unused (should all be 0)

-------------------------------------------------------------------------------
Table 1: Description of Machine Instructions
-------------------------------------------------------------------------------
Assembly language 	Opcode in binary		Action
name for instruction	(bits 24, 23, 22)
-------------------------------------------------------------------------------
add (R-type format)	000 			add contents of regA with
						contents of regB, store
						results in destReg.

nand (R-type format)	001			nand contents of regA with
						contents of regB, store
						results in destReg.

lw (I-type format)	010			load regB from memory. Memory
						address is formed by adding
						offsetField with the contents of
						regA.

sw (I-type format)	011			store regB into memory. Memory
						address is formed by adding
						offsetField with the contents of
						regA.

beq (I-type format)	100			if the contents of regA and
						regB are the same, then branch
						to the address PC+1+offsetField,
						where PC is the address of the
						beq instruction.

mult (R-type format)	101 			multiplies contents of regA with
						contents of regB, store
						results in destReg.

halt (O-type format)	110			increment the PC (as with all
						instructions), then halt the
						machine (let the simulator
						notice that the machine
						halted).

noop (O-type format)	111			do nothing.
-------------------------------------------------------------------------------

3. Requirements

This programming assignment requires the construction of a pipeline simulator,
written in C or C++, for the LC instruction set. Solutions to
this assignment will include the source code for the simulator, any test
programs used to verify correct program execution and a writeup (of about 2
pages) describing how the test program verify correct execution of any
legal program in the simulator.  Failure to provide the writeup, or failure
to provide a complete set suite will result in a lower grade, even if the
simulator correctly executes all test programs.

4. LC Pipelined Implementation

For this project we will use a datapath similar to the Pentium Pro discussed in
class.  Partial details of each pipeline stage follow:

Fetch: 1 cycle, fetch up to 2 instructions per cycle, use a bimodal branch
predictor with 64 entries (indexed by the least significant bits of the
instruction address) of 2-bit counters with states (taken, weakly-taken,
weakly-not-taken, and not-taken) and an inital state of weakly-not-taken for
each entry.  If the first instruction is a branch that is predicted taken,
send only 1 instruction to the next pipeline stage.  If both instructions
are branches, send only 1 instruction to the next pipeline stage.
You should implement a 3-entry fully associative branch target buffer (BTB)
in which all branches (taken or not taken) are entered, with LRU replacement.

Register rename:  Convert the architected register specifiers for each instruction
to physical register specifiers (using a rename table with 8 entries.  Update the
register mapping in the rename table for the destination register. Each entry
has a valid bit (0 means the current value in the register file is correct; 1
means that some instruction in the ROB is generating the required value) and
a rename index into the ROB for the instruction generating the required value.

Allocate: Place each instruction in the Reorder buffer and the correct
reservation station.  The reorder buffer contains 16 entries; 3-entry
reservation stations exist for 1) the add/nand/beq function unit, 2) the
multiply function unit, and 3) the memory unit which handles loads and stores
in program order (for a total of 9 reservation station entries).  noop and halt
instructions do not need a RS entry allocated.  If the reservation station or ROB
is full, stall all instructions in fetch, register rename and allocate.

Reservation station entries contain:
  src1 physical register identifier (index into 16 entry ROB)
  src2 physical register identifier (index into 16 entry ROB)
  dest physical register identifier (index into 16 entry ROB)
  src1 value (source operand value after reading or forwarding)
  src1 valid bit (says whether the src1 value is correct or still needs to be read)
  src2 value (source operand value after reading or forwarding)
  src2 valid bit (says whether the src1 value is correct or still needs to be read)
  operation to perform (for load/store and add/nand/beq RS)
  Any other storage required for correct function

Reorder buffer entries contain:
  Destination register ID (Architected register to update)
  Destination value
  Destination value valid bit (1 if the Destination value has been calculated)
  Any other storage required for correct function

Schedule: At most one instruction can be executed by each function unit each
cycle. Instrucions in the reservation station with all source values available
can be scheduled for execution.  If multiple instructions are available the
instruction in earliest program order (use the dest physical register specifier
and the head pointer in the ROB to determine age) is scheduled first, except
that beq's always have priority over adds and nands to reduce average branch
penalty.  Once an instruction is scheduled, the reservation station entry is
freed (it can be re-allocated in the same cycle it is freed).
  
Execute:  Adds, Nands and Beq take one cycle to execute.
Multiplies are pipelined and take 6 cycles to execute;  a new multiply with
all dependencies resolved can start executing 2 cycles after a previous
multiply has started (i.e., the multiply pipeline isn't completely pipelined).
Loads and stores take 3 cycles to execute and are not pipelined. To make things
easier, stores can update the memory contents at the end of execute (unlike
the Pentium Pro).  Once execution is complete, results are sent to the ROB (to
the index specified by the destination physical register specifier, which is
also the index into the ROB for that instruction) and broadcast to each
reservation station entry. One execution is complete, the reservation station
entry for that instruction is freed.

Commit:  Each cycle the two oldest entries in the reorder buffer can be
retired if they have completed execution.  If only the oldest instruction
has completed, then it can retire.  If the oldest entry has not completed, no
instructions can retire this cycle.  When retired, the ROB entry is freed,
the register rename table entry is updated (ONLY if it is still mapped to the 
retiring instruction), and the architected register file is updated.
Branch misprediction recovery occurs when the branch instruction is retired,
with ROB, reservation station and register rename tables cleared.


5. Problem

Your task is to write a cycle-accurate simulator for the LC.
At the start of the program, initialize the pc and all registers to zero.

run() will be a loop, where each iteration through the loop executes one cycle.
At the beginning of the cycle, print the complete state of the machine
In the body of the loop, you will figure out what the new state of the
machine (memory, registers, pipeline registers) will be at the end of the
cycle. 
Your simulation will halt when the halt instruction retires.

6. Output

Each cycle you should print out which instructions are in the ROB from newest to oldest.

At the end of execution you should print:

CYCLES:   cycle time to complete program (cycle when halt is in commit)
FETCHED:  # of instruction fetched (including instructions squashed because
          of branch misprediction)
RETIRED:  # of instruction committed
BRANCHES: # of branches executed (i.e., resolved)
MISPRED:  # of branches incorrectly predicted

7. Running Your Program

Your simulator should be run using the same command format specified in Project
1, that is:

	simulate program.mc > output

8. Test Cases

An integral (and graded) part of writing your pipeline simulator will be to
write a suite of test cases to validate any LC pipeline simulator.  This
is common practice in the real world--software companies maintain a suite of
test cases for their programs and use this suite to check the program's
correctness after a change.  Writing a comprehensive suite of test cases will
deepen your understanding of the project specification and your program, and
it will help you a lot as you debug your program.

The test cases for this project will be short assembly-language programs that,
after being assembled into machine code, serve as input to a simulator.  You
will submit your suite of test cases together with your simulator, and we will
grade your test suite according to how thoroughly it exercises an LC
pipeline simulator. 

The test cases for this simulator are significantly more complex than those
required for a simple inorder pipeline simulator.  Exhaustive testing in
nearly impossible, but end cases should be tested at the least.

9. Writeup

Finally you will produce a document describing the overall operation of your
simulator as well as a discussion of how each test case demonstrates correct
operation of some portion of your pipeline implementation.  This includes
executing each instruction type, correctly forwarding all data hazards and
correct operation of both the BTB and branch predictor.

The project writeup does not specify every aspect of the pipeline
organization.  Your writeup should explicitly state any design choices made
including a discussion of why you designed it and how it was tested. Failure
to provide this in the wrteup will severely degrade your score on the
assignment

10. Turning in the Project
