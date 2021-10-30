#include "In_Order_Pipeline_Simulator.h"

//Global variables
stateType state;	//current state of little computer
stateType newState; //new state of little computer
branchCtrlLineType branchCtrlLine; //represents the datapath that flows from the Mem stage to Instruction fetch for branch instructions
LittleComputer LittleCmprPipeLineSim; // Little computer PipeLine
branchTargetBufferType branchData; //holds the pc of a fetched instructions and target address for a taken branch

//BTB Code
bool branchTargetBufferEntryFound = false; 
std::list<branchTargetBufferType> branchTargetBuffer; //holds the the branch data of a taken branch. Up to 4 entrys
std::list<branchTargetBufferType>::const_iterator bufferEntryPtr; //points to the branch in the branch target buffer that was predicted last.
bool enableBit = false;
void main(int argc, char* argv[])
{
	string machineCodeFile  = argv[1];  
	string machineCodeLine;					//stores each line of assembly file
	ifstream machineFile(machineCodeFile); //open assembly file
	int address = 0;
	//open assemby language program file
	if (machineFile.is_open())
	{cout <<"intstruction memory: "<<endl;
		//read machineCodeLine from assembly language program file into string
		while ( getline(machineFile,machineCodeLine) )
		{
			// line from assembly file is blank
			if(!machineCodeLine.empty())
			{
				//When the program starts, read the machine-code file into BOTH instrMem and dataMem 
				state.instrMem[address] = atoi(machineCodeLine.c_str());
				state.dataMem[address]  = atoi(machineCodeLine.c_str());
				cout<<"instrMem["<<address<<"] "<<state.instrMem[address]<< '\n';
				printInstruction(atoi(machineCodeLine.c_str()));
				address++;
			}
			
		}//when done reading lines close file
		machineFile.close();
	}
	else
	{
		//can't open the assembly file
		cout << "Unable to open file: "<< machineCodeFile<<endl; 
	}
	state.numMemory = address; //set the number of memory addresses 
	
	run(); //begin pipeline simulation

	 exit(0);
}

void run()
{
	//initializing program counter and registers to zero prior to processing instructions
	state.pc				        	   = 0;
	LittleCmprPipeLineSim.fetched		   = 0;
	LittleCmprPipeLineSim.branchesTaken	   = 0;
	LittleCmprPipeLineSim.branchesMissPred = 0;

	//clear out all registers
	for(unsigned int i = 0; i < NUMREGS; i++)
	{
		state.reg[i] = 0;
	}
	//initializing instruction fields for pipeline registers
	state.IFID.instr  = NOOPINSTRUCTION;
	state.IDEX.instr  = NOOPINSTRUCTION;
	state.EXMEM.instr = NOOPINSTRUCTION;
	state.MEMWB.instr = NOOPINSTRUCTION;
	state.WBEND.instr = NOOPINSTRUCTION;
	state.MEMWB.squashInstruction = false;
	branchCtrlLine.branch = false; //branch control line data path is initially off since we know we can't check for a branch until the Mem stage.

	//Call queues: used to call each pipeline stage in the correct order. 
	std::queue<string> callQueue;      //stores each stage of the pipeline in order. Uses variables that represent each stage of the pipeline. 
	std::queue<string> newCallQueue;   //Used as temporary variable to hold the new stages that will need to be executed on the next clock cycle
	std::queue<string> emptyCallQueue; //Use to clear the newCallQueue at the end of the clock cycle

	bool cycleComplete = false; //determine when all the stages that need to execute during the current cycle have executed.

	while (1) 
	{//each iteration of this loops represents 1 clock cycle

		printState(&state);

		/* check for halt */
		if (opcode(state.MEMWB.instr) == HALT) {
			printf("machine halted\n");
			cout<<"CYCLES: "<<state.cycles<<endl;
			//FETCHED:  # of instruction fetched (including instructions squashed because of branch misprediction)
			cout<<"FETCHED: "<<LittleCmprPipeLineSim.fetched<<endl;
			//RETIRED:  # of instruction completed
			cout<<"RETIRED: "<<LittleCmprPipeLineSim.completed<<endl;
			//BRANCHES: # of branches executed (i.e., resolved)
			cout<<"BRANCHES: "<<LittleCmprPipeLineSim.branchesTaken<<endl;
			//MISPRED:  # of branches incorrectly predicted
			cout<<"MISPRED: "<<LittleCmprPipeLineSim.branchesMissPred<<endl;
			exit(0);
		}

		callQueue.push("IF"); //need to go through instruction fetch stage for current cycle

		newState = state;
		newState.cycles++;

		int queueSize = callQueue.size(); //store the size of the call queue for this clock cycle
		for(int i=0; i < queueSize; i++)
		{
			cout<<callQueue.front()<<endl;
			if(!callQueue.empty() && callQueue.front() == "IF" && !cycleComplete ) //IF stage needs to happen and the cycle is not complete
			{
				
				/* --------------------- IF stage --------------------- */
				LittleCmprPipeLineSim.IfIdStage(&newState);
				callQueue.pop(); //remove IF from stack since we just finished the IF/ID stage
				if(callQueue.empty()) //if this was the last stage that needed to process an instruction for this clock cycle
				{
					cycleComplete = true; //this clock cycle is finished
				}
				newCallQueue.push("ID"); //will need to decode instruction on next clock cycle
				LittleCmprPipeLineSim.fetched++;
			}
			if(!callQueue.empty() && callQueue.front() == "ID" && !cycleComplete )
			{
				/* --------------------- ID stage --------------------- */ 
				LittleCmprPipeLineSim.IdExStage(&newState);
				callQueue.pop();
				if(callQueue.empty()) //if this was the last stage that needed to process an instruction
				{
					cycleComplete = true; //this clock cycle is finished
				}
				newCallQueue.push("EX");
			}
			if(!callQueue.empty() && callQueue.front() == "EX" && !cycleComplete )
			{
				/* --------------------- EX stage --------------------- */
				LittleCmprPipeLineSim.ExMemStage(&newState);
				callQueue.pop();
				if(callQueue.empty()) //if this was the last stage that needed to process an instruction
				{
					cycleComplete = true; //this clock cycle is finished
				}
				newCallQueue.push("MEM");
			}
			if(!callQueue.empty() && callQueue.front() == "MEM" && !cycleComplete )
			{
				/* --------------------- MEM stage --------------------- */
				LittleCmprPipeLineSim.MemWbStage(&newState);
				callQueue.pop();
				if(callQueue.empty()) //if this was the last stage that needed to process an instruction
				{
					cycleComplete = true; //this clock cycle is finished
				}
				newCallQueue.push("WB");
			}
			if(!callQueue.empty() && callQueue.front() == "WB" && !cycleComplete )
			{
				/* --------------------- WB stage --------------------- */
				LittleCmprPipeLineSim.WbEndStage(&newState);
				callQueue.pop();
				if(callQueue.empty()) //if this was the last stage that needed to process an instruction
				{
					cycleComplete = true; //this clock cycle is finished
				}
				LittleCmprPipeLineSim.completed++;
			}	
		}
		cout<<endl<<endl;
		cycleComplete = false;			//time to begin a new cycle
		callQueue	  = newCallQueue;   //copy the new call stack to the active call queue
		newCallQueue  = emptyCallQueue; //clear previous new call queue for next clock cycle
		
		state = newState; /* this is the last statement before end of the loop.
					It marks the end of the cycle and updates the
					current state with the values calculated in this
					cycle */

    }
}

void LittleComputer::IfIdStage(stateType *newStateParam)
{
	if(enableBit == false)
	{

		newStateParam->IFID.instr    = newStateParam->instrMem[newStateParam->pc]; //fetch new instruction from memory based on program counter
		if(branchCtrlLine.branch == true) //if its time to branch
		{
			newStateParam->IFID.instr   = newStateParam->instrMem[newStateParam->EXMEM.branchTarget]; //get instruction that we branched to
			newStateParam->pc           = newStateParam->EXMEM.branchTarget+1;	//the new program counter is now our branch target address
			newStateParam->IFID.pcPlus1 = newStateParam->pc;	//the new program counter plus one is now our branch target address
			branchCtrlLine.branch = false;			//taken branch instruction has been used so reset the branch control line to false or "off"
			LittleCmprPipeLineSim.branchesTaken++;	//increment the branch taken counter for statistics.
			//branchTargetBufferEntryFound = false;
			return;
		}
		else
		{
			//Perform instruction fetch logic
			newStateParam->pc	         = newStateParam->pc + 1;					 //increment program counter
			newStateParam->IFID.pcPlus1  = newStateParam->pc;					    //store program counter for IF/ID pipeline register
			//branchTargetBufferEntryFound = false;
			return;
		}
	}


}
void LittleComputer::IdExStage(stateType *newStateParam)
{
	int regALocation = 0;
	int regBLocation = 0;

	//check for data hazards
	if(isDataHazard(getRegA(newStateParam->IFID.instr),newStateParam) || isDataHazard(getRegB(newStateParam->IFID.instr),newStateParam))
	{
		//stall cpu by setting enable bit so that noops are passed to execute
		enableBit = true;
		newStateParam->IDEX.instr = NOOP;
	}
	else
	{
		//continue execution

		//perform instruction decode logic
		if(newStateParam->MEMWB.squashInstruction == false)
		{
			newStateParam->IDEX.instr    = newStateParam->IFID.instr;	 //reads IFID pipeline register to get current instruction
		}
		else
		{
			newStateParam->MEMWB.squashInstruction = false;
		}
		newStateParam->IDEX.pcPlus1  = newStateParam->IFID.pcPlus1; //pass current program counter to ID/EX pipeline register
		newStateParam->IDEX.offset   = getDestReg(newStateParam->IFID.instr); //use the instruction in our pipeline register to get the destination register

		//Mux logic before decode instruction and read the registers based on instruction type
		if(getInstructionType(newStateParam->IFID.instr) == ITYPE)
		{	//initialize opcode, regA,regB, and offset for I-Type instructions

			//find out which register number regA and regB refer too
			regALocation = getRegA(newStateParam->IFID.instr);	//use the instruction in our pipeline register to get regA location
			regBLocation = getRegB(newStateParam->IFID.instr);	//use the instruction in our pipeline register to get regB location

			//read contents from register file or reg[NUMREGS]
			newStateParam->IDEX.readRegA = newStateParam->reg[regALocation];
			newStateParam->IDEX.readRegB = newStateParam->reg[regBLocation];
		}
		else if(getInstructionType(newStateParam->IFID.instr) == RTYPE)
		{ 		
			//find out which register number regA and regB refer too
			regALocation = getRegA(newStateParam->IFID.instr);   //use the instruction in our pipeline register to get regB location
			regBLocation = getRegB(newStateParam->IFID.instr);   //use the instruction in our pipeline register to get regB location
		
			//read contents from register file or reg[NUMREGS]
			newStateParam->IDEX.readRegA = newStateParam->reg[regALocation];
			newStateParam->IDEX.readRegB = newStateParam->reg[regBLocation];
		}
		else if(getInstructionType(newStateParam->IFID.instr) == OTYPE)
		{
			//noop or halt, nothing to read so initialize to zero
			newStateParam->IDEX.offset   = 0;
			newStateParam->IDEX.readRegA = 0;
			newStateParam->IDEX.readRegB = 0;
		}
	}


}
void LittleComputer::ExMemStage(stateType *newStateParam)
{
	int curOpcode = opcode(newStateParam->IDEX.instr); 
	//Mux logic: branch or ALU computation
	if( curOpcode == BEQ)
	{
		newStateParam->EXMEM.branchTarget = EXAlu(newStateParam,curOpcode);  //calculate branch target address
	}
	else
	{
		newStateParam->EXMEM.aluResult = EXAlu(newStateParam, curOpcode); //pass Register A and Register B to the Arithmetic logic Unit
	}

	newStateParam->EXMEM.instr    = newStateParam->IDEX.instr;

	//forward the contents of regB
	newStateParam->EXMEM.readRegB = newStateParam->IDEX.readRegB;

	if(curOpcode == NOOP)
	{
		newStateParam->EXMEM.branchTarget = 0;
		newStateParam->EXMEM.aluResult    = 0;
		newStateParam->EXMEM.readRegB     = 0;
	}

}
void LittleComputer::MemWbStage(stateType *newStateParam)
{
	int curOpcode = opcode(newStateParam->EXMEM.instr); //get opcode for this instruction
	if(curOpcode == BEQ && newStateParam->EXMEM.branchCtrlLineState == true) //if branch instruction and branch needs to be taken
	{
		branchCtrlLine.branch    = newStateParam->EXMEM.branchCtrlLineState; //set the current state of the branch ctrl line data path to "on" 
		branchData.targetAddress = newStateParam->EXMEM.branchTarget;
		//updateBTB(); // update the branch target buffer if necessary
		if(!branchTargetBufferEntryFound) //if the current branch instruction was NOT a predicted branch
		{
			branchTargetBuffer.push_back(branchData);  //update branch target buffer with the taken branches program counter and instruction
			//send control signals for noop instructions
			newStateParam->IDEX.instr  = NOOPINSTRUCTION;
			newStateParam->EXMEM.instr = NOOPINSTRUCTION;
			newStateParam->MEMWB.instr = NOOPINSTRUCTION;
			newStateParam->MEMWB.squashInstruction = true;
		}//else no stalls necessary
	}
	//if opcode is for a NAND instruction, forward alu result to mem/wb pipeline register
	newStateParam->MEMWB.aluResult = newStateParam->EXMEM.aluResult;
	//forward instruction bits
	newStateParam->MEMWB.instr     = newStateParam->EXMEM.instr;

	//determine if there is a read or write that needs to be done
    MemDataOperation(newStateParam,curOpcode);


}
void LittleComputer::WbEndStage(stateType *newStateParam)
{
	newStateParam->WBEND.writeData = newStateParam->MEMWB.writeData;
	newStateParam->WBEND.instr     = newStateParam->MEMWB.instr;
	newStateParam->WBEND.aluResult = newStateParam->MEMWB.aluResult;
	int curOpcode = opcode(newStateParam->WBEND.instr); //get opcode for this instruction
	int destinationRegister = 0;
	//Mux logic: ALU result or new data for register file
	if(curOpcode == LW)
	{
		destinationRegister = field1(newStateParam->WBEND.instr);  //use instruction bits to get destination register address
		//use the destination register which is bits 19-21 for I-type
		newStateParam->reg[destinationRegister] = newStateParam->WBEND.writeData;  //write back to register using effective address
	}
	else if(curOpcode == ADD || curOpcode == MULT || curOpcode == NAND)
	{
		destinationRegister = getDestReg(newStateParam->WBEND.instr); //use instruction bits to get destination register address
		//use the destination register which is first 2 bits of integer for R type
		newStateParam->reg[destinationRegister] = newStateParam->WBEND.aluResult;
	}
	
}

int LittleComputer::EXAlu(stateType *newStateParam,int curOpcode)
{
	
	if (curOpcode == ADD) //register to register ALU instruction
	{
		return newStateParam->IDEX.readRegB + newStateParam->IDEX.readRegA;   // add the contents of register A with the contents of register B
	} 
	else if (curOpcode == NAND) //register to register ALU instruction 
	{
		return ~(newStateParam->IDEX.readRegA & newStateParam->IDEX.readRegB);	//bitwise AND the offset with register A and then NOT the result
    }
	else if (curOpcode == LW || curOpcode == SW) 
	{   
		return newStateParam->IDEX.offset + newStateParam->IDEX.readRegA;    //form memory address by adding offset Field with contents of register A
    } 
	else if (curOpcode == BEQ)//register immediate
	{
		if(newStateParam->IDEX.readRegA == newStateParam->IDEX.readRegB)     //if register A and B contain the same value
		{
			//branch is valid
			newStateParam->EXMEM.branchCtrlLineState = true;
		}
		return newStateParam->IDEX.pcPlus1 + newStateParam->IDEX.offset; //add the program counter plus 1 adn the offset field to determine the target address to branch to
		
    } 
	else if (curOpcode == MULT) //register to register ALU instruction
	{
		return newStateParam->IDEX.readRegB * newStateParam->IDEX.readRegA;  //multiply the contents of register A with the contents of register B
    }
	else if (curOpcode == HALT)
	{
		return newStateParam->IDEX.pcPlus1 + newStateParam->IDEX.offset;
	}
	return 0;
}
int LittleComputer::MemDataOperation(stateType *newStateParam, int curOpcode)
{
	int effectiveAddress = newStateParam->EXMEM.aluResult;
	if(curOpcode == LW) //load word opcode so data must be read
	{
		newStateParam->MEMWB.writeData = newStateParam->dataMem[effectiveAddress]; //use the effective address to load data from memory
	}
	else if(curOpcode == SW) //store word opcode so data must be written
	{
		newStateParam->dataMem[effectiveAddress] =  newStateParam->EXMEM.readRegB; //write data read from register file in previous cycle to memory at the effective address
		//if(effectiveAddress > newStateParam->numMemory)
		//{
		//	newStateParam->numMemory = effectiveAddress+1;
		//}
	}
	else if(curOpcode == NOOP)
	{
		newStateParam->MEMWB.writeData = 0;
	}
	return 0;
}

bool LittleComputer::isDataHazard(int registerN,stateType *newStateParam)
{
	//get opCode for each instruction
	int exeOpcode,memOpcode,wbOpcode;
	exeOpcode = opcode(newStateParam->EXMEM.instr);
	memOpcode = opcode(newStateParam->MEMWB.instr);
	wbOpcode = opcode(newStateParam->WBEND.instr);

	//set the destination register for each stage
	int exeDest= -1,memDest = -1,wbDest = -1;
	if(opcode(newStateParam->IFID.instr) == NOOP || opcode(newStateParam->IFID.instr) == HALT)
	{
		return false;
	}

	if(exeOpcode != NOOP && exeOpcode != HALT )
	{
		if(exeOpcode == LW)
		{
			exeDest =  field1(newStateParam->EXMEM.instr);
		}
		else
		{
			exeDest = getDestReg(newStateParam->EXMEM.instr);
		}
	}

	if(memOpcode != NOOP && memOpcode != HALT )
	{
		if(memOpcode == LW)
		{
			memDest = field1(newStateParam->MEMWB.instr);
		}
		else
		{
			memDest = getDestReg(newStateParam->MEMWB.instr);
		}
	}

	if(wbOpcode != NOOP && wbOpcode != HALT )
	{
		if(wbOpcode == LW)
		{
			wbDest  = field1(newStateParam->WBEND.instr);
		}
		else
		{
			wbDest  = getDestReg(newStateParam->WBEND.instr);
		}

	}


	if(registerN == exeDest || registerN == memDest || registerN == wbDest)
	{
		return true;
	}

	return false;
}
//Utility functions
bool LittleComputer::branchTargetFound(int pc)
{
	return false;
}
int LittleComputer::getInstructionType(int instr)
{
	//the instruction type is based on the opcode so...
	int opcodeType = opcode(instr); //get opcode for this instruction
	//Mux logic: determine what the opcode is for the current instruction
	if(opcodeType == ADD || opcodeType == NAND || opcodeType == MULT)//opcode performs add, multiply and nand
	{
		//this is an R-type instruction
		return RTYPE;
	}
	else if(opcodeType == LW || opcodeType == SW || opcodeType == BEQ)//opcode perfroms load word, store word, or branch instruction
	{
		//this is an I-type instruction
		return ITYPE;
	}
	else if(opcodeType == HALT || opcodeType == NOOP)//opcode performs noop or halt 
	{
		//this is an O-type instruction
		return OTYPE;
	}
	return OTYPE;
} 
int LittleComputer::getRegA(int instr)
{
	return field0(instr);
}
int LittleComputer::getRegB(int instr)
{
	return field1(instr);
}

//Destination register and offset functions

short LittleComputer::getDestReg(int instr)
{
	return field2(instr);
}
short field2(int instruction)
{
    return(instruction & 0xFFFF);
}


void printState(stateType *statePtr)
{
    int i;
    printf("\n@@@\nstate before cycle %d starts\n", statePtr->cycles);
    printf("\tpc %d\n", statePtr->pc);

    printf("\tdata memory:\n");
	for (i=0; i<statePtr->numMemory; i++) {
	    printf("\t\tdataMem[ %d ] %d\n", i, statePtr->dataMem[i]);
	}
    printf("\tregisters:\n");
	for (i=0; i<NUMREGS; i++) {
	    printf("\t\treg[ %d ] %d\n", i, statePtr->reg[i]);
	}
    printf("\tIFID:\n");
	printf("\t\tinstruction ");
	printInstruction(statePtr->IFID.instr);
	printf("\t\tpcPlus1 %d\n", statePtr->IFID.pcPlus1);
    printf("\tIDEX:\n");
	printf("\t\tinstruction ");
	printInstruction(statePtr->IDEX.instr);
	printf("\t\tpcPlus1 %d\n", statePtr->IDEX.pcPlus1);
	printf("\t\treadRegA %d\n", statePtr->IDEX.readRegA);
	printf("\t\treadRegB %d\n", statePtr->IDEX.readRegB);
	printf("\t\toffset %d\n", statePtr->IDEX.offset);
    printf("\tEXMEM:\n");
	printf("\t\tinstruction ");
	printInstruction(statePtr->EXMEM.instr);
	printf("\t\tbranchTarget %d\n", statePtr->EXMEM.branchTarget);
	printf("\t\taluResult %d\n", statePtr->EXMEM.aluResult);
	printf("\t\treadRegB %d\n", statePtr->EXMEM.readRegB);
    printf("\tMEMWB:\n");
	printf("\t\tinstruction ");
	printInstruction(statePtr->MEMWB.instr);
	printf("\t\twriteData %d\n", statePtr->MEMWB.writeData);
    printf("\tWBEND:\n");
	printf("\t\tinstruction ");
	printInstruction(statePtr->WBEND.instr);
	printf("\t\twriteData %d\n", statePtr->WBEND.writeData);
}
int field0(int instruction)
{
    return( (instruction>>19) & 0x7);
}
int field1(int instruction)
{
    return( (instruction>>16) & 0x7);
}


int opcode(int instruction)
{
    return(instruction>>22);
}
void printInstruction(int instr)
{
    char opcodeString[10];
    if (opcode(instr) == ADD) {
	strcpy(opcodeString, "add");
    } else if (opcode(instr) == NAND) {
	strcpy(opcodeString, "nand");
    } else if (opcode(instr) == LW) {
	strcpy(opcodeString, "lw");
    } else if (opcode(instr) == SW) {
	strcpy(opcodeString, "sw");
    } else if (opcode(instr) == BEQ) {
	strcpy(opcodeString, "beq");
    } else if (opcode(instr) == MULT) {
	strcpy(opcodeString, "mult");
    } else if (opcode(instr) == HALT) {
	strcpy(opcodeString, "halt");
    } else if (opcode(instr) == NOOP) {
	strcpy(opcodeString, "noop");
    } else {
	strcpy(opcodeString, "data");
    }

    printf("%s %d %d %d\n", opcodeString, field0(instr), field1(instr),	field2(instr));
}

