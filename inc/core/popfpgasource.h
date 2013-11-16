#ifndef __POP_FPGA_SOURCE_HPP_
#define __POP_FPGA_SOURCE_HPP_



#include <core/popsink.hpp>
#include <core/popsource.hpp>

#include "PCIeAPI.h"         // the APIs
#include "LSCPCIe2_IF.h"     // the driver interface
#include "GPIO.h"            // the GPIO IP module
#include "SFIF.h"            // the SFIF IP module
#include "MiscUtils.h"
#include "MemFormat.h"
#include "SFIF.h"            // the SFIF IP module
#include "MemMap.h"       // IP module base addresses

using namespace std;
using namespace LatticeSemi_PCIe;

namespace pop
{


class PopFPGASource
{
private:

	string boardName;
	unsigned boardNum;
	string demoName;
	LSCPCIe2_IF *pDrvr;
	SFIF        *pSFIF;
	GPIO        *pGPIO;

	uint32_t WriteDmaBuf[FIFO_SIZE/4];
	uint32_t ReadDmaBuf[FIFO_SIZE/4];

	PCIeAPI theDLL;

public:

	PopFPGASource():boardName("ECP3"), boardNum(1), demoName("Basic"),
	pDrvr(0), pSFIF(0), pGPIO(0)

	{

	}

	~PopFPGASource()
	{
		if( pDrvr )
			delete pDrvr;
		if( pSFIF )
			delete pSFIF;
		if( pGPIO )
			delete pGPIO;
	}


	/**
	 * Display the SFIF TLP counter registers.
	 * Usually called after a run to see what has transpired - number of TLPs
	 * sent, number CplD recvd, elapsed time, etc.
	 * @param sfifCnts the SFIF structure defining the various register fields
	 */
	void    showSFIFCounters(SFIF::SFIFCntrs_t &sfifCnts)
	{
		cout<<"ELAPSED_CNT: SFIF[0x14] = "<<dec<<sfifCnts.ElapsedTime <<endl;
		cout<<"TX_TLP_CNT: SFIF[0x18] = "<<sfifCnts.TxTLPCnt<<endl;
		cout<<"RX_TLP_CNT: SFIF[0x1C] = "<<sfifCnts.RxTLPCnt<<endl;
		cout<<"CREDIT_WR_PAUSE_CNT: SFIF[0x20] = "<<sfifCnts.WrWaitTime<<endl;
		cout<<"LAST_CPLD_TS: SFIF[0x24] = "<<sfifCnts.LastCplDTime<<endl;
		cout<<"CREDIT_RD_PAUSE_CNT: SFIF[0x28] = "<<sfifCnts.RdWaitTime<<endl;
	}



	void    readTlpMenu(void)
	{
		uint32_t cycles;
		uint32_t icg;
		uint32_t nPkts;
		uint32_t readReqSize;
//		uint32_t numDWs;
		uint64_t totalBytes;
		char line[80];
		double   pauseRatio;
		double   thruput;
		string outs;

		SFIF::SFIFCntrs_t sfifCnts;


		while (1)
		{

			cout<<"\n\n";
			cout<<"==================================================\n";
			cout<<"  Read a TLP(s) from system memory\n";
			cout<<"==================================================\n";
			cout<<"Read Request Size(4,16,32,64,128,256,512): ";
			//cin>>readReqSize;
			readReqSize = 16;
			switch (readReqSize)
			{
			case 4:
			case 16:
			case 32:
			case 64:
			case 128:
			case 256:
			case 512:
//				numDWs = readReqSize / 4;
				break;
			default:
				readReqSize = 4;
//				numDWs = 1;
			}

//			cout<<"Number MRd TLPs to send: ";
//			cin>>nPkts;
			nPkts = 1;
			if (nPkts > 32)
			{
				cout<<"NOTE: Max 32 outstanding TLPs."<<endl;
				nPkts = 32;
			}
			cout<<"# Cycles of MRd pkts: ";
//			cin>>cycles;
			cycles = 1;
			if (cycles >= 0x10000)
			{
				cout<<"NOTE: Max 64k cycles."<<endl;
				cycles = 0xffff;
			}
			else if (cycles < 1)
				cycles = 1;

			cout<<"ICG: ";
//			cin>>icg;

			// how many clocks to wait between reads
			icg = 0;

			// Put pattern to read in DMA common buffer done by SFIF class

			pSFIF->setupSFIF(READ_CYC_MODE,	 // runMode
					MRD_TLPS,		  // trafficMode
					cycles,
					icg,
					readReqSize,  // rdTLPSize
					0,			   // wrTLPSize
					nPkts,				// numRdTLPs
					0);   // numWrTLPs


			pSFIF->startSFIF(false);  // not looping, just run the set many cycles

			Sleep(1000);

			pSFIF->stopSFIF();


			pSFIF->getCntrs(sfifCnts);
			showSFIFCounters(sfifCnts);


			//pauseRatio = (double)sfifCnts.RdWaitTime / (double)sfifCnts.ElapsedTime;
			pauseRatio = (double)sfifCnts.RdWaitTime / (double)sfifCnts.LastCplDTime;
			cout<<"Run-time waiting for MRd credits: "<<(pauseRatio * 100.0)<<"%\n";

			if (pSFIF->getSFIFParseRxFIFO(outs))
			{
				cout<<outs;
			}
			else
			{
				cout<<"\n\nThruput estimate based on read request size, Rx TLP count and Last CplD time\n";
				if (readReqSize < RCB)
					totalBytes = (uint64_t)readReqSize * (uint64_t)sfifCnts.RxTLPCnt;
				else
					totalBytes = (uint64_t)RCB * (uint64_t)sfifCnts.RxTLPCnt;
				thruput = (totalBytes / (8E-9 * sfifCnts.LastCplDTime)) / 1E6;

				cout<<"Recvd: "<<dec<<totalBytes<<" bytes  in "<<(8E-9 * sfifCnts.LastCplDTime)<<" secs   Throughput: "<<thruput<<" MB/s\n";
			}


//			cout<<"\n\nAgain(y/n)? ";
//			cin>>line;
//			if (line[0] != 'y')
			break;
		}

	}









	void init()
	{
		string infoStr;
		uint32_t rd32_val;
		uint32_t ui;
		int i;
		char line[80];

			cout<<    "======================================\n";
			cout<<"Dll version info: "<<theDLL.getVersionStr()<<endl;
			// Setup so lots of diag output occurs from tests
				theDLL.setRunTimeCtrl(PCIeAPI::VERBOSE);























				try
				{
					cout<<"\n\n======================================\n";
					cout<<    "       Driver Interface Info\n";
					cout<<    "======================================\n";
					cout<<"Opening LSCPCIe2_IF...\n";
					pDrvr = new LSCPCIe2_IF(boardName.c_str(),
											demoName.c_str(),
											boardNum);

					cout<<"OK.\n";

					pDrvr->getDriverVersionStr(infoStr);
					cout<<"Version: "<<infoStr<<endl;
					pDrvr->getDriverResourcesStr(infoStr);
					cout<<"Resources: "<<infoStr<<endl;
					pDrvr->getPCICapabiltiesStr(infoStr);
					cout<<"Link Info: "<<infoStr<<endl;


					//==================================================
					//  lscpcie2 Driver Info Test
					//==================================================
					cout<<"\n\n";
					cout<<"==========================================\n";
					cout<<"  Step #1: get PCIe resources\n";
					cout<<"==========================================\n";

					const PCIResourceInfo_t *pInfo;
					pDrvr->getPCIDriverInfo(&pInfo);
					cout<<"lscpcie Driver Info:\n";
					cout<<"numBARs = "<<pInfo->numBARs<<endl;
					for (ui = 0; ui < MAX_PCI_BARS; ui++)
					{
						if (ui >= pInfo->numBARs)
							cout<<"***";  // not initialized
						cout<<"\tBAR"<<ui<<":";
						cout<<"  nbar="<<pInfo->BAR[ui].nBAR;
						cout<<"  type="<<(int)pInfo->BAR[ui].type;
						cout<<"  size="<<pInfo->BAR[ui].size;
						cout<<"  Addr="<<hex<<pInfo->BAR[ui].physStartAddr<<dec;
						cout<<"  mapped="<<pInfo->BAR[ui].memMapped;
						cout<<"  flags="<<pInfo->BAR[ui].flags<<endl;
					}
					cout<<"hasInterrupt="<<pInfo->hasInterrupt<<endl;
					cout<<"intrVector="<<pInfo->intrVector<<endl;


					pDrvr->getPCIExtraInfoStr(infoStr);
					cout<<"\n\nDevice Driver Info:\n"<<infoStr<<endl;

					const ExtraResourceInfo_t *pExtra;
					pDrvr->getPciDriverExtraInfo(&pExtra);

					if (pExtra->DmaBufSize < FIFO_SIZE)
					{
						cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
						cout<<"            ERROR\n";
						cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
						cout<<"\nDMA Common Buffer smaller than SFIF TX FIFO\n";
						cout<<"Exiting to avoid possible buffer overflows.\n";
						exit(-1);
					}

					if (pExtra->DmaPhyAddrLo % 4096)
					{
						cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
						cout<<"WARNING!  DMA Common Buffer Base is not 4kB aligned.\n";
						cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
						cout<<"\nNo support to handle crossing 4kB address boundary\n";
						cout<<"Continue? (y/n): ";
						cin>>line;
						if (line[0] != 'y')
							exit(-1);
					}

					if (pExtra->DmaPhyAddrLo % 128)
					{
						cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
						cout<<"WARNING!  Base is not 128 aligned.\n";
						cout<<"Writes need to account for crossing 4k boundary\n";
						cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
						cout<<"\nNo support to handle crossing 4kB address boundary\n";
						cout<<"Continue? (y/n): ";
						cin>>line;
						if (line[0] != 'y')
							exit(-1);
					}


					//==================================================
					//  lscpcie2 Driver DMA Buf access test
					//==================================================
					cout<<"\n\n";
					cout<<"=======================================================\n";
					cout<<"  Step #2: Test Read/Write DMA Sys buf from User Space\n";
					cout<<"=======================================================\n";

					cout<<"Testing accessing DMA Sys Buf from User space"<<endl;
					cout<<"Filling WriteBuf with 0xa5\n";
					for (i = 0; i < 1024; i++)
						WriteDmaBuf[i] = (i<<16) | i;
					cout<<"Calling driver to write DMA common buf\n";
					pDrvr->writeSysDmaBuf(WriteDmaBuf, 4096);

					cout<<"Clearing ReadBuf\n";
					memset(ReadDmaBuf, 0x00, 4096);
					cout<<"Calling driver to read DMA common buf\n";
					pDrvr->readSysDmaBuf(ReadDmaBuf, 4096);
					for (i = 0; i < 1024; i++)
					{
						if (ReadDmaBuf[i] != WriteDmaBuf[i])
							cout<<"ERROR! DMA Read != Write"<<endl;
					}
					cout<<"PASS\n";



					//==================================================
					// Access SFIF
					//==================================================
					cout<<"\n\n";
					cout<<"=======================================================\n";
					cout<<"  Step #4: Create SFIF device object and test\n";
					cout<<"=======================================================\n";

					pSFIF = new SFIF("SFIF",		 // a unique name for the IP module instance
									 memSFIF(0),	// its base address
									 pDrvr);		// driver interface to use for register access

					// Display all the SFIF register values for visual check
					pSFIF->showRegs();

					cout<<"\nWrite ICG_Count = 0xbeef\n";
					pSFIF->write32(0x08, 0xbeef);
					cout<<"Read TX_ICG: SFIF[8] = ";
					rd32_val = pSFIF->read32(0x08);
					cout<<hex<<rd32_val<<dec<<endl;
					if (rd32_val != 0xbeef)
					{
						cout<<"ERROR! Did not read back what was written!\n";
					}
					else
					{
						cout<<"PASS\n";
					}

					cout<<endl << endl << endl << endl;

					readTlpMenu();

//
//
//					bool done = false;
//					while (!done)
//					{
//						cout<<"\n\n";
//						cout<<"==================================================\n";
//						cout<<"            SFIF Menu\n";
//						cout<<"==================================================\n";
//						cout<<"R = (Read) MRd TLP Menu\n";
//						cout<<"W = (Write) MWr TLP Menu\n";
//						cout<<"T = Thruput Tests\n";
//						cout<<"G = GPIO Module Tests\n";
//						cout<<"S = SFIF Register Status\n";
//						cout<<"P = PCIe Link Settings\n";
//						cout<<"C = Change TLP Configuration Settings\n";
//						cout<<"D = Display contents of DMA Common buf\n";
//						cout<<"x = Exit\n";
//						cout<<"-> ";
//						cin>>line;
//
//						switch (line[0])
//						{
//							case 'r':
//							case 'R':
//								readTlpMenu();
//								break;
//
//							case 'w':
//							case 'W':
//								writeTlpMenu();
//								break;
//
//							case 'g':
//							case 'G':
//								GPIOTest();
//								break;
//
//							case 't':
//							case 'T':
//								thruputTests();
//								break;
//
//							case 'd':
//							case 'D':
//								cout<<"\nSystem Common Buffer Contents Display";
//								pDrvr->readSysDmaBuf(ReadDmaBuf, FIFO_SIZE);
//								cout<<hex;
//								for (i = 0; i < FIFO_SIZE/4; i++)
//								{
//									if ((i % 4) == 0)
//										printf("\n%04X:  ", i*4);
//									printf("%08X  ", ReadDmaBuf[i]);
//								}
//								cout<<dec;
//								break;
//
//							case 'c':
//							case 'C':
//								changeConfigSettings();
//								break;
//
//							case 'p':
//							case 'P':
//								//showPCIeSettings(pWBM);
//								pDrvr->getPCIResourcesStr(infoStr);
//								cout<<infoStr<<endl;
//								pDrvr->getPCICapabiltiesStr(infoStr);
//								cout<<infoStr<<endl;
//								pDrvr->getPCIExtraInfoStr(infoStr);
//								cout<<infoStr<<endl;
//								break;
//
//							case 's':
//							case 'S':
//								cout<<"\n\n";
//								cout<<"============================\n";
//								cout<<"  Read SFIF Registers\n";
//								cout<<"============================\n";
//								pSFIF->showRegs();
//
//								cout<<"\nRoot Complex Initial Credits:"<<endl;
//								cout<<"PD_CA (Wr): "<<((pGPIO->read32(0x24))>>16)<<endl;
//								cout<<"NPH_CA(Rd): "<<((pGPIO->read32(0x24)) & 0xffff)<<endl;
//								break;
//
//							case 'x':
//							case 'X':
//								done = true;
//								break;
//						}
//					}

				}
				catch (std::exception &e)
				{
					cout<<"\n!!!ERROR!!! Testing Halted with errors: "<<e.what()<<endl;
				}























	}



};

}




#endif
