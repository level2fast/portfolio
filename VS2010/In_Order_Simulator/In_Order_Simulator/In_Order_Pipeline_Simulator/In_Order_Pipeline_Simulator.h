#include <iostream>
#include <fstream>
#include <string>
#include <queue>       
#include <list>
using namespace std;

#define NOOPINSTRUCTION 0x1c00000
#define NUMMEMORY 65536 /* maximum number of data words in memory */
#define NUMREGS 8 /* number of machine registers */
#define ADD 0  
#define NAND 1
#define LW 2
#define SW 3
#define BEQ 4
#define MULT 5
#define HALT 6
#define NOOP 7
#define RTYPE 8
#define ITYPE 9
#define OTYPE 10
#define INITIAL_STATE 0

typedef struct IFIDStruct { //instruction fetch/instruction decode
    int instr;   //instruction
    int pcPlus1; //program counter +1
} IFIDType;

typedef struct IDEXStruct {//instruction decode/execute
    int instr;     //instruction
    int pcPlus1;   //program counter +1
    int readRegA;  //register A -temporary register used to hold the value that rs refers too
    int readRegB;  //Register B -temporary register used to hold the value that rt refers too
    int offset;    ///offest or sign extended immediate 
} IDEXType;

typedef struct EXMEMStruct {
    int instr;         //instruction
    int branchTarget;  //memory address to branch to
    int aluResult;     //output of Arithmetic Logic Unit calculation
    int readRegB;      //value read from Register B
	bool branchCtrlLineState;
} EXMEMType;

typedef struct MEMWBStruct {
    int instr;       //instruction
    int writeData;   //data that will be written to register
	int aluResult;   //results of arithmetic logical unit operation
	bool squashInstruction;
} MEMWBType;

typedef struct WBENDStruct {
	int aluResult;  //results of arithmetic logical unit operation
    int instr;      //instruction register that holds current instruction
    int writeData;  //data that will be written to register
} WBENDType;

typedef struct stateStruct {
    int pc;                    //program counter register
    int instrMem[NUMMEMORY];   //instruction memory cache
    int dataMem[NUMMEMORY];    //instruction data cache
    int reg[NUMREGS];          //registers
    int numMemory;             //
    IFIDType IFID;             //instruction fetch/instruction decode state   
    IDEXType IDEX;             //instruction decode/execute state
    EXMEMType EXMEM;           //execute/memory state  
    MEMWBType MEMWB;           //memory/writeback state
    WBENDType WBEND;           //writeback/end state
    int cycles;				   //number of cycles run so far
} stateType;

typedef struct branchCtrLineStruct {//this structure represents the control line that would move data from the EX stage to the Mux in the IF stage
		 bool branch;	//input to mux, determines whether or no the LC will branch
		 int branchControlLineData; //data sent through control line when branch is true
} branchCtrlLineType;


void printInstruction(int instr);
void printState(stateType *statePtr);
int field0(int instruction);

int field1(int instruction);

short field2(int instruction);
int   field2_int(int instruction);


int opcode(int instruction);//End LC Assembly Language and Assembler

void main(int argc, char* argv[]);
void run();

typedef struct branchTargetBuffer {
		 int pc;			//program counter of instruction to fetch
		 int targetAddress; //predicted program counter
} branchTargetBufferType;
class LittleComputer
{
	public:
		void IfIdStage(stateType *newStateParam);
		void IdExStage(stateType *newStateParam);
		void ExMemStage(stateType *newStateParam);
		void MemWbStage(stateType *newStateParam);
		void WbEndStage(stateType *newStateParam);
		bool isDataHazard(int registerN,stateType *newStateParam);
		int getOffSet(int instr);
		short getDestReg(int instr);
		int getRegA(int instr);
		int getRegB(int instr);
		int getInstructionType( int instr);
		int EXAlu( stateType *newStateParam, int curOpcode);
		int EXMux(int var1, int var2);
		int MemReadOrWrite(stateType *newState);
		int MemDataOperation(stateType *newState, int curOpcode);
		bool branchTargetFound(int pc);
		int fiveCycleCounter;
	    int curOpcode;
		int fetched;
		int completed;
		int branchesTaken;
		int branchesMissPred;


};