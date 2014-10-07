#include <iostream>
#include <boost/timer.hpp>
#include <algorithm>    // std::sort
#include <boost/lexical_cast.hpp>

#include "core/poppackethandler.hpp"
#include "core/util.h"
#include "core/popartemisrpc.hpp"
#include "core/basestationfreq.h"
#include "core/utilities.hpp"
#include "core/popchannelmap.hpp"
#include "dsp/prota/popsparsecorrelate.h"
#include "b64/b64.h"


using namespace std;

namespace pop
{



#define QUICK_SEARCH_STEPS (3000)
#define DATA_SAMPLE(x) data[x]

// how good of a match is required to attempt demodulate
#define COMB_COORELATION_FACTOR ((double)0.10)


uint32_t pop_correlate_spool(const uint32_t* data, const size_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut, uint32_t* finalSample, uint32_t endPadding)
{
	uint32_t denseCombLength = comb[combSize-1] - comb[0];
	uint32_t denseDataLength = 0;

	// the best score possible (100% correlation) is equal to the length of the comb
	uint32_t threshold = COMB_COORELATION_FACTOR * denseCombLength;

	uint32_t i;

	// we are forced to scan through the input data to determine if any modulus events have occurred in order to get a real value for denseDataLength
	for(i = 1; i < dataSize; i++)
	{
		if( DATA_SAMPLE(i) < DATA_SAMPLE(i-1) )
		{
			//denseDataLength += ARTEMIS_CLOCK_SPEED_HZ;
			printf("bump (%d)\r\n", i);
		}

		denseDataLength += DATA_SAMPLE(i)-DATA_SAMPLE(i-1);
	}

	if( denseDataLength < (denseCombLength+endPadding) )
	{
//		printf("dense data size %"PRIu32" must not be less than dense comb size %"PRIu32"\r\n", denseDataLength, denseCombLength);

		*scoreOut = 0;

		//FIXME: this is not an appropriate way of returning an error condition
		return 0;
	}

	int32_t score, scoreLeft, scoreRight, maxScoreQuick = 0, maxScore = 0; //x(key)score
	uint32_t maxScoreOffsetQuick, maxScoreOffset, scoreOffsetBinSearch, iterations, combOffset;


	// Artemis is given a "guess" of the start timer value when the start-of-frame should occur
#ifdef POPWI_PLATFORM_ARTEMIS
	iterations = guess - DATA_SAMPLE(0) + GUESS_ERROR + 1;
	combOffset = guess - DATA_SAMPLE(0) - GUESS_ERROR;
#else
	iterations = denseDataLength - denseCombLength + 1;
	combOffset = 0;
#endif

	std::vector<uint32_t> matchOffsets;
	std::vector<int32_t> matchScores;

	int64_t lastOffset = -1;

	int count = 0;

	uint32_t state = 0;

	// quick search
	for(; combOffset < iterations; combOffset += QUICK_SEARCH_STEPS)
	{
		score = do_comb(data, dataSize, comb, combSize, combOffset, &state);

		// if the score passes the threshold
		if( abs(score) > threshold )
		{
			// if we found two consecutive matches (this must be the same packet)
			if( lastOffset != -1 )
			{
				// if this match is better than the last one
				if( abs(score) > abs(matchScores.back()) )
				{
					// remove previous score, and use this one
					matchOffsets.pop_back();
					matchScores.pop_back();
					matchOffsets.push_back(combOffset);
					matchScores.push_back(score);
				}
			}
			else
			{
				// this is the first match above the threshold in awhile, save it
				matchOffsets.push_back(combOffset);
				matchScores.push_back(score);
			}


			// remember the offset of the last match
			lastOffset = combOffset;
		}
		count++;
	}

	// this is the dense value of the last "quick" search we tried
	uint32_t finalDense = combOffset-QUICK_SEARCH_STEPS+data[0];

	uint32_t samp=0;

	for( i = 0; i < dataSize; i++ )
	{
		if( data[i] < finalDense )
		{
			samp = i;
		}
	}

	*finalSample = samp;



//	cout << "got " << matchOffsets.size() << " thresholded combs (" << count << ")" << endl;

	uint32_t ret = 0;

	if( matchOffsets.size() > 1 )
	{
		cout << "got MULTIPLE different packets" << endl;
	}

	for(i = 0; i < matchOffsets.size(); ++i)
	{

//		cout << "climbing match " << i <<  " offset: " << matchOffsets[i] << " score: " << abs(matchScores[i]) << endl;
//		cout << "thresh: " << threshold << endl;




		// we've found a peak
		uint32_t searchStep = QUICK_SEARCH_STEPS;

		maxScoreOffset = scoreOffsetBinSearch = matchOffsets[i];
		maxScore = matchScores[i];


		// warmup loop; we only need to do a single comb because the previous one was done in the quick search
		scoreRight = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch+1, &state);

		if( abs(maxScoreQuick) > abs(scoreRight) )
		{
			scoreOffsetBinSearch -= searchStep/2;
		}
		else
		{
			scoreOffsetBinSearch += searchStep/2;
		}


		while( searchStep != 1 )
		{
			searchStep /= 2;

			scoreLeft = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch, &state);

			scoreRight = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch+1, &state);

			if( abs(scoreLeft) > abs(scoreRight) )
			{
				if( abs(scoreLeft) > abs(maxScore) )
				{
					maxScore = scoreLeft;
					maxScoreOffset = scoreOffsetBinSearch;
				}

				scoreOffsetBinSearch -= searchStep/2;
			}
			else
			{
				if( abs(scoreRight) > abs(maxScore) )
				{
					maxScore = scoreRight;
					maxScoreOffset = scoreOffsetBinSearch+1;
				}

				scoreOffsetBinSearch += searchStep/2;
			}

			if( searchStep == 1 && scoreLeft == scoreRight )
			{
				//FIXME: this condition can be fixed by curve fitting the searched spots
				printf("Flat peak detected, start of frame will be slightly wrong\r\n");
			}
		}




//		printf("max scoreee: %d\r\n", maxScore);


		if( i == 0 )
		{
			//FIXME
			ret = DATA_SAMPLE(0) + maxScoreOffset;

			*scoreOut = maxScore;

		}





	}


	return ret;


//	printf("max: %u %d\r\n", maxScoreOffsetQuick, maxScoreQuick);



//	printf("Max offset bin:   %u\r\n", maxScoreOffset);

	//*scoreOut = maxScore;

//	return DATA_SAMPLE(0) + maxScoreOffset;
}









uint32_t comb[] = {0,4,9,15,23,31,34,40,41,42,49,56,57,61,67,68,69,73,79,86,92,98,104,111,119,125,132,135,139,146,151,155,158,161,169,170,177,179,187,194,201,206,207,212,214,221,229,231,234,241,247,250,257,262,265,271,273,281,287,291,297,299,306,309,311,319,327,335,342,349,352,359,367,368,374,376,377,384,389,390,393,394,396,399,405,406,408,411,413,419,420,422,427,431,432,437,445,449,450,454,462,464,466,472,475,479,481,487,488,494,496,500,504,507,513,520,523,527,533,538,543,549,556,560,567,570,575,578,581,586,594,599,600,605,610,613,614,618,626,633,637,639,647,650,656,663,665,672,675,677,684,686,694,696,697,700,706,712,718,719,720,724,726,727,732,734,742,748,751,757,763,766,771,772,779,780,787,793,799,807,808,809,810,816,824,829,830,831,833,837,839,843,847,855,858,862,869,875,880,883,891,893,894,900,902,903,905,912,915,921,928,930,935,940,941,947,953,954,957,962,970,974,977,982,986,992,995,997,1004,1010,1014,1016,1020,1022,1026,1028,1033,1040,1042,1049,1051,1056,1062,1068,1069,1072,1076,1084,1089,1097,1101,1107,1114,1121,1126,1127,1128,1133,1141,1146,1150,1154,1155,1158,1160,1161,1163,1164,1170,1175,1183,1184,1190,1198,1200,1203,1210,1217,1225,1226,1233,1240,1244,1249,1252,1257,1263,1269,1273,1281,1283,1287,1293,1301,1303,1305,1311,1316,1318,1322,1323,1328,1330,1336,1339,1347,1351,1352,1359,1363,1367,1371,1378,1384,1385,1393,1396,1399,1403,1405,1408,1410,1412,1420,1424,1426,1433,1440,1447,1452,1457,1462,1467,1475,1481,1489,1491,1497,1502,1506,1512,1516,1524,1528,1530,1531,1539,1541,1549,1555,1558,1559,1563,1568,1573,1575,1581,1583,1589,1590,1593,1599,1605,1610,1617,1618,1625,1631,1637,1639,1645,1650,1651,1653,1655,1658,1659,1665,1666,1673,1679,1681,1682,1689,1696,1701,1703,1711,1719,1727,1728,1731,1732,1734,1738,1740,1748,1752,1753,1755,1758,1759,1765,1771,1772,1779,1780,1783,1786,1793,1801,1806,1807,1814,1815,1819,1822,1823,1827,1832,1833,1841,1848,1853,1861,1868,1872,1878,1881,1885,1889,1890,1893,1901,1904,1911,1913,1920,1923,1929,1931,1936,1937,1943,1948,1955,1962,1966,1969,1977,1983,1989,1994,1997,2005,2011,2019,2027,2030,2038,2045,2051,2059,2066,2070,2075,2076,2079,2084,2089,2095,2098,2106,2111,2116,2117,2118,2125,2129,2136,2140,2147,2151,2158,2166,2167,2170,2171,2175,2177,2185,2193,2194,2201,2206,2207,2210,2211,2214,2218,2226,2233,2235,2240,2247,2250,2252,2258,2261,2263,2270,2277,2278,2281,2288,2290,2296,2301,2307,2310,2318,2324,2326,2330,2338,2341,2349,2352,2353,2354,2355,2357,2362,2365,2366,2370,2371,2373,2376,2382,2387,2391,2392,2394,2396,2403,2405,2413,2421,2429,2437,2443,2446,2453,2457,2465,2466,2467,2474,2482,2484,2487,2490,2491,2499,2501,2504,2508,2509,2517,2521,2528,2530,2537,2542,2550,2555,2558,2566,2569,2571,2576,2580,2586,2593,2595,2597,2604,2611,2614,2617,2624,2631,2632,2640,2648,2650,2651,2659,2661,2664,2672,2673,2676,2680,2682,2683,2685,2689,2690,2696,2698,2704,2706,2710,2711,2714,2717,2719,2723,2731,2734,2740,2741,2749,2751,2757,2765,2770,2773,2779,2780,2786,2788,2793,2800,2807,2815,2822,2828,2831,2837,2844,2851,2856,2862,2866,2872,2873,2878,2879,2880,2882,2885,2890,2893,2897,2901,2902,2904,2912,2914,2918,2920,2926,2930,2938,2940,2945,2949,2950,2953,2960,2963,2966,2974,2981,2983,2987,2995,2999,3001,3005,3011,3014,3017,3021,3026,3032,3035,3038,3040,3048,3055,3062,3066,3068,3069,3071,3073,3076,3083,3090,3094,3096,3097,3102,3108,3112,3115,3117,3119,3123,3130,3138,3142,3143,3144,3148,3152,3153,3156,3164,3167,3175,3178,3180,3186,3193,3198,3199,3204,3206,3210,3218,3222,3224,3226,3229,3235,3236,3242,3249,3256,3257,3263,3267,3268,3271,3272,3275,3283,3289,3291,3295,3302,3306,3311,3317,3318,3322,3326,3330,3336,3342,3347,3350,3354,3355,3361,3369,3372,3374,3382,3389,3395,3401,3404,3409,3410,3411,3417,3422,3430,3436,3444,3445,3453,3457,3458,3466,3467,3469,3473,3476,3481,3483,3486,3487,3493,3500,3504,3508,3516,3521,3525,3532,3533,3536,3543,3547,3550,3555,3561,3566,3569,3570,3578,3579,3583,3589,3597,3603,3604,3612,3614,3619,3622,3625,3628,3635,3640,3646,3654,3655,3658,3660,3662,3667,3675,3676,3684,3688,3689,3693,3695,3696,3699,3700,3702,3704,3705,3710,3718,3720,3722,3728,3729,3731,3733,3736,3739,3740,3743,3744,3749,3752,3759,3762,3768,3772,3773,3776,3777,3782,3783,3789,3792,3799,3806,3807,3814,3819,3821,3823,3824,3827,3835,3838,3844,3849,3856,3858,3865,3871,3874,3879,3881,3882,3890,3892,3900,3906,3909,3912,3919,3926,3930,3932,3935,3937,3942,3946,3953,3958,3961,3963,3967,3975,3976,3983,3988,3995,4000,4006,4007,4015,4016,4019,4027,4032,4037,4039,4040,4046,4053,4061,4066,4073,4075,4083,4084,4092,4094,4096,4103,4108,4114,4121,4127,4133,4137,4140,4141,4147,4151,4155,4161,4167,4168,4173,4175,4179,4183,4184,4190,4198,4206,4208,4214,4215,4220,4228,4230,4236,4240,4241,4245,4251,4256,4262,4264,4268,4275,4279,4282,4290,4294,4300,4308,4312,4315,4322,4323,4325,4327,4331,4338,4341,4349,4352,4354,4362,4370,4375,4383,4385,4388,4396,4399,4404,4406,4413,4420,4425,4429,4430,4432,4437,4445,4453,4460,4462,4465,4471,4479,4484,4487,4490,4494,4500,4504,4512,4518,4526,4530,4533,4535,4536,4543,4546,4550,4557,4564,4571,4576,4579,4586,4590,4592,4600,4604,4612,4613,4620,4628,4632,4640,4644,4650,4654,4659,4664,4665,4667,4672,4673,4676,4679,4681,4683,4690,4691,4696,4704,4711,4717,4719,4720,4721,4722,4726,4734,4740,4748,4752,4755,4758,4765,4773,4778,4782,4784,4792,4799,4805,4808,4811,4813,4815,4821,4823,4825,4830,4836,4842,4843,4851,4859,4867,4870,4875,4876,4878,4886,4889,4896,4902,4910,4912,4913,4919,4923,4931,4937,4940,4944,4947,4953,4954,4962,4968,4972,4974,4977,4983,4988,4990,4997,5000,5007,5008,5013,5018,5026,5028,5035,5036,5041,5045,5052,5056,5063,5071,5076,5081,5082,5087,5091,5093,5095,5097,5103,5105,5112,5120,5126,5133,5136,5142,5145,5146,5154,5162,5168,5174,5178,5180,5187,5192,5199,5207,5211,5214,5221,5224,5227,5233,5236,5239,5241,5248,5251,5258,5264,5271,5273,5281,5283,5288,5293,5299,5302,5307,5309,5311,5317,5322,5326,5331,5335,5342,5350,5357,5358,5365,5372,5376,5381,5383,5390,5393,5401,5402,5408,5413,5415,5416,5420,5428,5436,5441,5447,5449,5451,5458,5459,5467,5469,5471,5479,5482,5488,5493,5494,5497,5505,5511,5513,5520,5522,5528,5534,5540,5544,5548,5553,5560,5564,5568,5572,5574,5581,5583,5585,5592,5598,5606,5614,5615,5616,5624,5629,5630,5633,5641,5645,5653,5659,5666,5672,5674,5678,5679,5684,5691,5692,5697,5698,5702,5706,5711,5716,5721,5722,5725,5727,5731,5737,5744,5752,5757,5765,5772,5774,5775,5782,5786,5790,5798,5806,5807,5810,5812,5813,5820,5823,5830,5832,5840,5845,5853,5861,5864,5866,5871,5875,5882,5888,5890,5891,5892,5897,5904,5905,5909,5915,5923,5926,5929,5934,5938,5946,5950,5955,5961,5962,5968,5976,5979,5981,5988,5990,5998,6001,6008,6009,6014,6019,6027,6032,6034,6035,6043,6046,6054,6056,6058,6064,6068,6076,6083,6085,6093,6094,6100,6107,6109,6117,6119,6125,6126,6132,6133,6141,6143,6145,6152,6157,6163,6169,6174,6179,6181,6188,6191,6199,6203,6208,6212,6217,6220,6227,6228,6233,6241,6245,6253,6255,6261,6269,6271,6272,6273,6277,6282,6290,6291,6299,6302,6308,6310,6314,6319,6321,6325,6329,6334,6336,6338,6345,6351,6353,6358,6366,6372,6374,6381,6387,6390,6393,6400,6406,6414,6420,6423,6427,6428,6436,6441,6444,6446,6447,6449,6452,6460,6468,6472,6479,6486,6488,6495,6501,6506,6510,6515,6521,6526,6528,6530,6538,6543,6548,6551,6558,6559,6565,6571,6575,6582,6585,6590,6595,6601,6602,6606,6609,6612,6616,6618,6625,6629,6633,6637,6639,6647,6654,6662,6663,6671,6678,6679,6686,6691,6695,6703,6704,6709,6716,6718,6721,6724,6730,6734,6740,6744,6752,6756,6760,6763,6767,6770,6776,6778,6779,6782,6784,6789,6790,6797,6799,6806,6808,6809,6812,6820,6824,6831,6834,6838,6840,6841,6845,6852,6855,6862,6864,6865,6872,6878,6882,6884,6890,6897,6898,6904,6911,6918,6924,6927,6928,6933,6938,6939,6943,6946,6952,6953,6957,6963,6967,6972,6973,6976,6984,6992,6996,6998,6999,7002,7005,7009,7014,7022,7029,7035,7042,7049,7050,7056,7057,7058,7063,7065,7067,7068,7074,7080,7081,7082,7087,7092,7099,7100,7104,7106,7108,7112,7117,7123,7129,7133,7141,7143,7148,7150,7152,7154,7161,7162,7165,7172,7180,7186,7188,7193,7194,7202,7204,7210,7212,7218,7224,7230,7234,7239,7240,7241,7243,7249,7251,7259,7262,7268,7272,7274,7281,7286,7287,7291,7292,7298,7305,7312,7320,7325,7329,7336,7344,7349,7356,7363,7369,7376,7377,7381,7386,7388,7393,7399,7402,7405,7411,7414,7418,7420,7425,7426,7434,7441,7446,7453,7456,7459,7460,7465,7467,7470,7477,7485,7489,7496,7498,7503,7504,7512,7516,7519,7526,7534,7537,7542,7546,7552,7556,7558,7564,7571,7576,7580,7585,7590,7592,7596,7600,7608,7614,7619,7626,7629,7631,7637,7643,7650,7658,7664,7668,7672,7674,7681,7686,7687,7689,7692,7693,7699,7707,7712,7719,7720,7723,7729,7733,7734,7738,7746,7752,7756,7761,7764,7769,7771,7773,7781,7783,7791,7795,7800,7805,7809,7812,7818,7826,7828,7829,7832,7835,7840,7847,7854,7859,7861,7863,7864,7870,7874,7877,7880,7886,7893,7901,7903,7908,7912,7914,7916,7919,7927,7933,7935,7938,7944,7948,7956,7961,7964,7971,7976,7982,7987,7995,7999};
uint32_t comb_end[] = {0, 3, 6, 9, 11, 12, 14, 16, 20, 23, 28, 32, 33, 38, 39, 43, 47, 52, 56, 57, 61, 65, 68, 72, 77, 82, 87, 89, 92, 97, 100, 105, 109, 114, 118, 120, 124, 126, 128, 130, 134, 137, 140, 141, 142, 142, 146, 147, 148, 150, 155, 156, 160, 163, 166, 169, 171, 173, 174, 176, 178, 183, 183, 188, 190, 192, 197, 197, 200};





PopPacketHandler::PopPacketHandler(unsigned notused) : PopSink<uint32_t>("PopPacketHandler", 3000), rpc(0), new_timers(0), artemis_tpm_start(-1), rework_start(0), rework_score(0)
{
	ldpc = new LDPC();
	ldpc->parse_mat2str();

	// do these before combs are modified in place
	start_comb_bits = comb[ARRAY_LEN(comb)-1];
	end_comb_bits = comb_end[ARRAY_LEN(comb_end)-1];

	size_t i;
	for(i = 0; i < ARRAY_LEN(comb); i++)
	{
		comb[i] = comb[i] * COUNTS_PER_BIT;
	}
	for(i = 0; i < ARRAY_LEN(comb_end); i++)
	{
		comb_end[i] = comb_end[i] * COUNTS_PER_BIT;
	}
}

// sort by uuid first, then by time
bool PopPacketQueueCompare(const PopPacketHandler::PopPacketQueue& lhs, const PopPacketHandler::PopPacketQueue& rhs)
{
	int comparison;

	comparison = lhs.uuid.compare(rhs.uuid);

	if( comparison < 0 )
	{
		return true;
	}
	else if( comparison > 0 )
	{
		return false;
	}
	else
	{
		return lhs.time.get_real_secs() < rhs.time.get_real_secs();
	}
}

void PopPacketHandler::enqueue_packet(std::string to, ota_packet_t& packet)
{
	cout << "enqueued packet to " << to << endl;
	PopPacketQueue q;

	q.time = get_microsec_system_time();
	q.uuid = to;
	q.packet = packet;

	queue.push_back(q);

	// this is very expensive but fun!
	std::sort (queue.begin(), queue.end(), PopPacketQueueCompare);


//	for (std::vector<PopPacketQueue>::iterator it = queue.begin(); it!=queue.end(); ++it)
//	{
//		std::cout << ' ' << it->time << ' ' << it->uuid << ' ' << it->packet.data << endl;
//	}
}

// Next packet to be transmitted (requires queue to already be sorted)
// Returns null if no packets are pending
ota_packet_t* PopPacketHandler::peek_packet(std::string uuid)
{
	for (std::vector<PopPacketQueue>::iterator it = queue.begin(); it!=queue.end(); ++it)
	{
		if( uuid.compare(it->uuid) == 0 )
		{
			return &(it->packet);
		}
//		std::cout << ' ' << it->time << ' ' << it->uuid << ' ' << it->packet.data << endl;
	}

	// no packets waiting
	return NULL;
}

// Deletes a packet from the queue
void PopPacketHandler::erase_packet(std::string uuid, ota_packet_t& packet)
{
	for (std::vector<PopPacketQueue>::iterator it = queue.begin(); it!=queue.end(); ++it)
	{
		if( uuid.compare(it->uuid) == 0 && memcmp(&packet, &(it->packet), sizeof(ota_packet_t) ) == 0)
		{
			queue.erase(it);
			return;
		}
	}
}

// returns the nearest slot for a tracker
int32_t PopPacketHandler::pop_get_nearest_slot(uuid_t uuid, int32_t slot_in)
{
	std::vector<PopChannelMap::PopChannelMapKey> keys;
	std::vector<PopChannelMap::PopChannelMapValue> values;
	map->find_by_tracker(uuid, keys, values);

	int32_t diff = POP_SLOT_COUNT + 1; // worst case
	uint32_t diff_slot = 0;

	int32_t tmp;


	//	cout << "System Slot: " << system_now_slot << endl;
	//	cout << "Closest slot: " << pop_get_tracker_slot_now(uuid) << endl;

	for( unsigned i = 0; i < keys.size(); i++ )
	{
		const PopChannelMap::PopChannelMapKey& key = keys[i];
		PopChannelMap::PopChannelMapValue val = values[i];

		tmp = abs((int32_t) key.slot - slot_in);

		if( tmp < diff )
		{
			diff = tmp;
			diff_slot = key.slot;
		}

		// now compare against wrapped version
		tmp = abs((int32_t) key.slot + POP_SLOT_COUNT - slot_in);

		if( tmp < diff )
		{
			diff = tmp;
			diff_slot = key.slot;
		}
	}

	if( diff == POP_SLOT_COUNT + 1 )
	{
		cout << "something seriously wrong pop_get_tracker_slot_now (was s3p restarted?)" << endl;
		return 0;
	}

	cout << "Slot: " << diff_slot << " is closest to system's now slot: " << slot_in << endl;

	return diff_slot;
}

// takes the "now" timeslot from the system clock and looks through the the timeslot list for device to find the closest slot
// this should be the timeslot that the tracker thinks it is transmitting on
int32_t PopPacketHandler::pop_get_tracker_slot_now(uuid_t uuid)
{
	// according to basestation clock, what slot is it?
	PopTimestamp system_now = get_microsec_system_time();
	uint64_t system_pit = round(system_now.get_frac_secs() * 19200000.0) + system_now.get_full_secs()*19200000;
	int32_t system_now_slot = pop_get_slot_pit_rounded(system_pit);

	return pop_get_nearest_slot(uuid, system_now_slot);
}

int PopPacketHandler::basestation_should_respond(uuid_t uuid)
{
	std::vector<PopChannelMap::PopChannelMapKey> keys;
	std::vector<PopChannelMap::PopChannelMapValue> values;
	map->get_full_map(keys, values);

	std::vector<std::string> basestations;

	// build vector of unique basestation names
	for( unsigned i = 0; i < keys.size(); i++ )
	{
		PopChannelMap::PopChannelMapValue val = values[i];
		if( std::find(basestations.begin(), basestations.end(), val.basestation) == basestations.end() )
		{
			basestations.push_back(val.basestation);
		}
	}

	uint16_t count = basestations.size();
	uint16_t crc = crcSlow(uuid.bytes, sizeof(uuid.bytes)) >> 3; // crcSlow doesn't seem to ever give odd results?

//	cout << "crc " << crc <<  endl;

	// modulus crc of device serial by count of basestations
	uint16_t result = crc % count;

	cout << "crc " << crc << " result " << result << " bs is " << basestations[result] << endl;

	// determine if this basestation should reply to the packet
	if( basestations[result].compare(pop_get_hostname()) == 0 )
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void PopPacketHandler::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], char *str, uint32_t txTime, uint64_t pitTxTime, uint64_t pitPrnCodeStart)
{
	std::string method = FROZEN_GET_STRING(methodTok);
	const struct json_token *params, *p0, *p1, *p2;

	int32_t original_id = -1;

	if( idTok )
	{
		original_id = parseNumber<int32_t>(FROZEN_GET_STRING(idTok));
	}


	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
//			rcp_log(FROZEN_GET_STRING(p0));
//			respond_int(0, methodId);
		}
	}

	if( method.compare("utc_rq") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			uuid_t uuid = b64_to_uuid(FROZEN_GET_STRING(p0));
			cout << "Serial: " << FROZEN_GET_STRING(p0) << endl;


			if( !basestation_should_respond(uuid) )
			{
				cout << "NOT REPLYING" << endl;
				return;
			}


			// Traditionally all ota packets are replied with a "tx" rpc here.
			// but in this case we want the basestation to behave smart, so we actually send an rpc

			int32_t original_id = -1;

			if( idTok )
			{
				original_id = parseNumber<int32_t>(FROZEN_GET_STRING(idTok));
			}

			char buf[128];

			// we encapsulate the original rpc id so that the basestation can correctly reply
			snprintf(buf, 128, "{\"method\":\"bs_send_utc_reply\",\"params\":[%d, %d, %ld]}", original_id, txTime, pitTxTime);

			printf("\r\n");
			puts(buf);

			rpc->send_rpc(buf, strlen(buf));
		}
	}

	if( method.compare("slot_rq") == 0  && original_id != -1 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			uuid_t uuid = b64_to_uuid(FROZEN_GET_STRING(p0));

			if( !basestation_should_respond(uuid) )
			{
				cout << "NOT REPLYING" << endl;
				return;
			}


			//cout << "slot rq from " << FROZEN_GET_STRING(p0) << endl;

			rpc->fabric->add_name(FROZEN_GET_STRING(p0));



			// how many slots are we giving out?
			unsigned remaining = 5;
			unsigned chosen = 0;
			uint16_t slots[remaining];

			// grab all slots available to us
			std::vector<PopChannelMap::PopChannelMapKey> keys;
			std::vector<PopChannelMap::PopChannelMapValue> values;
			map->find_by_basestation(pop_get_hostname(), keys, values);


			int walk = 6;
			int offset = -1;

			//		unsigned n = keys.size() ; // size before the inserts
			for( unsigned i = 0; i < keys.size(); i++ )
			{
				const PopChannelMap::PopChannelMapKey& key = keys[i];
				PopChannelMap::PopChannelMapValue val = values[i];
//				PopChannelMap::PopChannelMapValue updatedVal = val;


				if( val.tracker == zero_uuid || val.tracker == uuid ) // give slot to tracker if it's empty OR if we've already given it to the same tracker
				{
					if( offset == -1 )
					{
						offset = key.slot;
					}

					if( ((key.slot-offset) % walk) != 0 )
					{
						continue;
					}


					slots[chosen] = key.slot;
					chosen++;

					map->set(key.slot, uuid, val.basestation);


					if( chosen >= remaining )
					{
						break;
					}
				}
			}

			ota_packet_t packet;
			ota_packet_zero_fill(&packet);

			ostringstream os;
			os << "{\"result\":[";
			for( unsigned i = 0; i < chosen; i++ )
			{
				if( i != 0 )
				{
					os << ",";
				}
				os << slots[i];
			}
			os << "],\"id\":" << original_id << "}";

			snprintf(packet.data, sizeof(packet.data), "%s", os.str().c_str()); // lazy way to cap length
			ota_packet_prepare_tx(&packet);

			puts(packet.data);

			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);

		}

//
//		for (std::vector<mystruct>::iterator iter = Vect.begin(); iter != Vect.end(); ++iter)
//		{
//			Vect.insert(iter + 1, otherstruct);
//
//		}

	}

	if( method.compare("poll") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		p1 = find_json_token(arr, "params[1]");
		if( p0 && p0->type == JSON_TYPE_STRING && p1 && p1->type == JSON_TYPE_NUMBER )
		{
			std::string uuid_string = FROZEN_GET_STRING(p0);
			uuid_t uuid = b64_to_uuid(uuid_string);

			if( !basestation_should_respond(uuid) )
			{
				cout << "NOT REPLYING" << endl;
				return;
			}


			double pit_epoc = (double)pitPrnCodeStart/19200000.0;

			cout << "start: " << pitPrnCodeStart << endl;
//			printf("Epoc: %lf\r\n", pit_epoc);

			uint64_t tracker_pit = parseNumber<uint64_t>(FROZEN_GET_STRING(p1));

			cout << "BS PIT: " << pitPrnCodeStart << endl;
			cout << "T  PIT: " << tracker_pit << endl;
//			cout << "t slot: " <<

			// this trim includes jitter between the time that the tracker builds the packet with it's PIT value and the time when it's transmitted.
			// A better way is to determine which slot the tracker was trying to tx, calculate the pit counts which differ from that slot, and correct based on that
//			int64_t error_counts = tracker_pit - pitPrnCodeStart;

			uint32_t target_slot = pop_get_nearest_slot( uuid, pop_get_slot_pit_rounded(tracker_pit) );
			int64_t error_counts = pop_get_slot_error(target_slot, pitPrnCodeStart);



//			PopTimestamp system_now = get_microsec_system_time();
//			uint64_t system_pit = round(system_now.get_frac_secs() * 19200000.0) + system_now.get_full_secs()*19200000;

			cout << "target_slot: " << target_slot << endl;
			cout << "error_counts: " << error_counts << endl;
//			cout << "System Slot: " << pop_get_slot_pit_rounded(system_pit) << endl;
//			cout << "Closest slot: " << pop_get_tracker_slot_now(uuid) << endl;

			ota_packet_t packet;
			ota_packet_zero_fill(&packet);

			ota_packet_t* queued_packet;

			// check if there is anything queued up
			queued_packet = peek_packet(uuid_string);


			if( queued_packet )
			{
				memcpy(&packet, queued_packet, sizeof(ota_packet_t));
				erase_packet(uuid_string, *queued_packet);
			}
			else if( abs(error_counts) > 0.05 * 19200000.0 )
			{
				ostringstream os;
				os << "{\"method\":\"trim_utc\",\"params\":[" << -1*error_counts << "]}";

				snprintf(packet.data, sizeof(packet.data), "%s", os.str().c_str()); // lazy way to cap length

				cout << endl;
				puts(packet.data);
				cout << endl << endl;
			}
			else
			{
				// do nothing
				snprintf(packet.data, sizeof(packet.data), "{}");
			}






			ota_packet_prepare_tx(&packet);

			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
		}
	}
}

// most of this was copied from popjsonrpc.cpp
void PopPacketHandler::process_ota_packet(ota_packet_t* p, uint32_t txTime, uint64_t pitTxTime, uint64_t pitPrnCodeStart)
{
	const char *json = p->data;

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok = 0, *paramsTok = 0, *idTok = 0;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		cout << "problem with json string (" <<  json << ")" << endl;
		return;
	}

	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
	{
		cout << "problem with json string (too many things for us to parse)" << endl;
		return;
	}

	// verify message has "method" key
	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		return;
	}

	// verify message has "params" key
	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		return;
	}

	// "id" key is optional.  It's absence means the message will not get a response
	idTok = find_json_token(arr, "id");
	if( !(idTok && idTok->type == JSON_TYPE_NUMBER) )
	{
		idTok = 0;
	}

	execute(methodTok, paramsTok, idTok, arr, p->data, txTime, pitTxTime, pitPrnCodeStart);
}


void PopPacketHandler::process(const uint32_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	// from here below we are reading timer values
	boost::mutex::scoped_lock lock(timer_mtx);


	static uint32_t total_samples = 0;
	total_samples += size;

//	if( total_samples > 6000000 )
//	{
//		total_samples = 0;
//		std::string msg = "{\"method\":\"tmr_sync\",\"params\":[]}";
//		rpc->send_rpc(msg);
//	}

	if( total_samples < 1500*10 )
	{
		cout << "got " << size << " samples" << endl;
	}

	uint32_t combDenseLength = comb[ARRAY_LEN(comb)-1];


	size_t i,j;
	uint32_t prnCodeStart, prnCodeStartUnscaled;
	uint32_t prnEndCodeStart;

	int32_t scorePrn, scoreBitSync;

	boost::timer t; // start timing


	static uint32_t previous_run_offset = 0;

	// was was the # of the last sample we took
	uint32_t final_sample = 0;

	data -= previous_run_offset;
	size += previous_run_offset;

	//cout << "processing " << size << " samples" << endl;


	if( new_timers == 0 )
	{
		return;
	} else if( artemis_tpm_start == -1 )
	{
//		for(i = 0; i < size;i++)
//		{
//			if( data[i] > artemis_tpm )
//			{
//				artemis_tpm_start = 0;
//				previous_run_offset = size-i;
//				return;
//			}
//		}
//
//		// Havent hit start condition yet
//		artemis_tpm_start = 0;
//		previous_run_offset = 0;
//		return;
	}





	for(i = 1; i<size;i++)
	{
		// the data wraps
		if( data[i-1] > data[i] )
		{
			// nothing can handle a wrap yet, so just bail and try again
			cout << "Data wrap edge condition" << endl;

			previous_run_offset = size-i;
			return;
		}
	}


	uint32_t end_padding = 0;

    // if we found a comb in the previous process, but didn't have enough data to move forward
    // the comb start is already saved
	if( rework_start == 0 )
	{
		prnCodeStartUnscaled = prnCodeStart = pop_correlate_spool(data, size, comb, ARRAY_LEN(comb), &scorePrn, &final_sample, end_padding);
	} else
	{
		prnCodeStartUnscaled = prnCodeStart = rework_start;
		scorePrn = rework_score;
	}



	int temp = 0;

	if( temp )
	{
		for(j = 0; j<size; j++)
		{
			printf("%u, ", data[j]);
		}
	}



//	printf("Score: %d\r\n", scorePrn);
//	printf("Start: %d\r\n", prnCodeStart);
//	if( abs(scorePrn) < )

	double elapsed_time = t.elapsed();

//	if( prnCodeStart == 0 || elapsed_time > 4.0 )
//	{
//		printf("\r\n");
//
//		for(j = 0; j<size-1;j++)
//		{
//			printf("%u, ", data2[j]);
//		}
//
////		pop_correlate(data2, size-1, comb, ARRAY_LEN(comb), &scorePrn);
//
//	}

//	printf("time %f\r\n", elapsed_time);



	// keep the artemis_tpm, artemis_pit, artemis_pps counters no more than 4 seconds behind
//	for(i = 0; i < size; i++)
//	{
//		//uint32_t mod = data[i] - artemis_tpm;
//		if( ((uint32_t)(data[i] - artemis_pps)) > (ARTEMIS_CLOCK_SPEED_HZ*4) )
//		{
////			cout << "    bump from: " << artemis_tpm << " to " << (artemis_tpm + ARTEMIS_CLOCK_SPEED_HZ) << " to data[" << i << "]: " << data[i] << endl;
//			//cout << "  mod: " << mod << endl;
//
//			//cout << "bump with: " << data[i] << endl;
//
//			// bump all the counters
//			artemis_tpm += ARTEMIS_CLOCK_SPEED_HZ;
//			artemis_pit += ARTEMIS_PIT_SPEED_HZ;
//			artemis_pps += ARTEMIS_CLOCK_SPEED_HZ;
//			artimes_pps_full_sec++;
//		}
//
//	}

	//cout << "sec: " << artimes_pps_full_sec << endl;





	if( prnCodeStart != 0 )
	{

		uint32_t end_padding_tight, first_sample_padding;


		uint32_t message_bits = 176 * 100;


		end_padding_tight = COUNTS_PER_BIT*(start_comb_bits+message_bits+end_comb_bits);
		first_sample_padding = 8*COUNTS_PER_BIT;

		// if we've found a comb, but there aren't enough samples ahead of us to represent the entire message
		if( (data[size-1] - prnCodeStart) < end_padding_tight )
		{
			uint32_t samp=0;
			cout << "rework" << endl;

			rework_start = prnCodeStart;
			rework_score = scorePrn;

			// find the edge right before the comb
//			for(i = 0; i < size; i++)
//			{
//				if( data[i] < (prnCodeStart - first_sample_padding) )
//				{
//					samp = i;
//				}
//			}

			previous_run_offset = size - samp;
			return;

		}

		// we've got enough data in our buffer, reset these flags
		rework_start = rework_score = 0;

		printf("Score: %d (%g)\r\n", scorePrn, abs(scorePrn)/(double)comb[ARRAY_LEN(comb)-1]);
		cout << "prnCodeStart: " << prnCodeStart << endl;


		int32_t scorePrnEnd, prnEndCodeStartUnscaled;
		uint32_t final_sample_burn = 0;
		uint32_t data_scaled[size];

		double scale_factor = 1;//(double)expected_length / (double)ldpcDataLength;

		// do end once outside of loop (because we already did beginning once outside of loop)
		prnEndCodeStartUnscaled = prnEndCodeStart = pop_correlate_spool(data, size, comb_end, ARRAY_LEN(comb_end), &scorePrnEnd, &final_sample_burn, 0);

		if( prnEndCodeStartUnscaled == 0 )
		{

//			cout << endl;
//			for(i = 0; i < size; i++)
//			{
//				cout << data[i] << ",";
//			}
//			cout << endl;

			cout << "trying again because we couldn't find final comb" << endl;


			previous_run_offset = size - final_sample_burn;
			return;
		}


		uint32_t ldpcDataStart, ldpcDataLength, expected_length;

		cout << "scaled prnCodeStart: " << prnCodeStart << endl;
		cout << "scaled prnEndCodeStart: " << prnEndCodeStart << endl;
		ldpcDataStart = prnCodeStart+combDenseLength;
		ldpcDataLength = prnEndCodeStart - ldpcDataStart;
		cout << "scaled ldpcDataLength: " << ldpcDataLength << endl;


		// prep
		//		memcpy( data_scaled, data, sizeof(uint32_t) * size);

		for( i = 0; i < 1; i++ )
		{

			expected_length = counts_per_bits(17600);

			scale_factor *= (double)expected_length / (double)ldpcDataLength;

			cout << "factor: " << boost::lexical_cast<string>(scale_factor) << endl;

			//		memcpy( data_scaled, data, sizeof(uint32_t) * size);
			for(j = 0; j < size; j++)
			{
				data_scaled[j] = round(data[j] * scale_factor);
			}


			prnCodeStart = pop_correlate_spool(data_scaled, size, comb, ARRAY_LEN(comb), &scorePrn, &final_sample, end_padding);
			prnEndCodeStart = pop_correlate_spool(data_scaled, size, comb_end, ARRAY_LEN(comb_end), &scorePrnEnd, &final_sample_burn, 0);


			cout << "scaled prnCodeStart: " << prnCodeStart << endl;
			cout << "scaled prnEndCodeStart: " << prnEndCodeStart << endl;
			ldpcDataStart = prnCodeStart+combDenseLength;
			ldpcDataLength = prnEndCodeStart - ldpcDataStart;
			cout << "scaled ldpcDataLength: " << ldpcDataLength << endl;

			cout << endl;
			cout << endl;

		}

		bool scale_factor_ok = 0;

		if (fabs(scale_factor - 1.0) < 0.1)
		{
			scale_factor_ok = 1;
		}




		uint32_t alc = ARRAY_LEN(comb);
		uint32_t comb2[alc];




		uint32_t artemis_tpm;
		uint64_t artemis_pit;
		uint32_t artemis_pps;
		uint64_t artimes_pps_full_sec;

		int flag = 0;
		for (auto timer = artemis_timers.rbegin(); timer != artemis_timers.rend(); ++timer)
		{
			boost::tie(artemis_tpm, artemis_pit, artemis_pps, artimes_pps_full_sec) = *timer;
//			cout << "pps: " << prnCodeStart - artemis_pps << endl;
			if( (prnCodeStart - artemis_pps) < 48000000 )
			{
//				cout << "yes";
				flag = 1;
				break;
			}
		}

		if( flag == 0 )
		{
			cout << "couldn't find pps near enough!!" << endl;
		}



//		cout << endl;
//
//		for(i = 0; i < size; i++)
//		{
//			cout << data[i] << ",";
//		}
//
//		cout << endl;




		uint32_t delta_counts = prnCodeStartUnscaled - artemis_tpm;

		if( delta_counts > ARTEMIS_CLOCK_SPEED_HZ )
		{
			cout << "    delta_counts: " << delta_counts << endl;
		}


//		 artemis_tpm << " pit: " << artemis_pit << " pps: " << artemis_pps
		// timers are syncronized which gives maching values at the same time
		// (tpm start of frame - last synced tpm) over tpm period times pit period = delta pit counts
		uint64_t pitPrnCodeStart = ( (delta_counts) / 48000000.0 ) * 19200000.0;

		// offset to get absolute counts
		pitPrnCodeStart += artemis_pit;

		cout << "pitPrnCodeStart: " << pitPrnCodeStart << endl;

//		double pit_epoc = (double)pitPrnCodeStart/19200000.0;
//		static double pit_epoc_last;

//		printf("PIT start: %lf\r\n", pit_epoc);
//		printf("PIT delta: %lf\r\n", pit_epoc-pit_epoc_last);


//		pit_epoc_last = pit_epoc;

		double txDelta = 1.7;



		// add delta seconds
		uint32_t txTime = (prnCodeStart + (uint32_t)(ARTEMIS_CLOCK_SPEED_HZ*txDelta));

		uint64_t pitTxTime = pitPrnCodeStart + (uint32_t)(ARTEMIS_PIT_SPEED_HZ*txDelta);







		//cout << "pit delta: " << ( pitTxTime - pitPrnCodeStart ) << endl;











		ota_packet_t rx_packet;

//		unsigned peek_length = 4; // how many bytes do we need to find the size?
//		unsigned peek_length_encoded = ota_length_encoded(peek_length);

//		uint8_t data_rx[peek_length_encoded];

//		int p = 0;
		size_t n = 17600;
		size_t k = 176;



		size_t ldpc_ota_bytes = n/8;

		uint8_t data_ldpc[ldpc_ota_bytes];

//		int bpered = 2640;
//		bpered /= 3;

		uint32_t guess_start = prnCodeStart+combDenseLength;

		cout << endl << "Guess ldpc data start:  " << guess_start << endl;
//		guess_start = 2509215548;



//		for( p = -2*bpered; p < 2*bpered; p+=bpered/16 )
//		{
//			if( p == 0 )
//			{
//				cout << " " << endl;
//			}
//			shannon_pop_data_demodulate(data, size, guess_start + p, data_ldpc, peek_length_encoded, (scorePrn<0?1:0));
//
//			ostringstream os;
//			os << hex << (int)data_ldpc[0] << ", " << (int)data_ldpc[1] << ", " << (int)data_ldpc[2] << "  (" << guess_start+p << ")"  << endl;
//
//			cout << os.str();
//		}

//		shannon_pop_data_demodulate(data, size, guess_start, data_ldpc, peek_length_encoded, (scorePrn<0?1:0));
//
//		{
//		ostringstream os;
//		os << hex << (int)data_ldpc[0] << ", " << (int)data_ldpc[1] << ", " << (int)data_ldpc[2] << "  (" << guess_start+p << ")"  << endl;
//		cout << os.str();
//		}



		uint16_t llr_data_size = n;
		int16_t llr_data[llr_data_size];

		core_pop_llr_demodulate(data_scaled, size, guess_start, llr_data, llr_data_size, (scorePrn<0?1:0));

//		for( i = 0; i < llr_data_size; i++ )
//		{
//			cout << llr_data[i] << endl;
//		}

//		cout << endl;


		if( scale_factor_ok )
		{
			ldpc->run(llr_data, llr_data_size);
		}
		else
		{
			// we correlated against noise for the start comb, and the end comb correlated different place than expected
			// meaning this isn't a valid packet
			cout << "data scale factor looks too far off" << endl;
		}






//		while(1) {}



//		uint8_t data_decode[peek_length];
//
//		uint32_t data_decode_size;
//
//		decode_ota_bytes(data_rx, peek_length_encoded, data_decode, &data_decode_size);
//
//
//		//	printf("data: %02x\r\n", data_decode[0]);
//		//	printf("data: %02x\r\n", data_decode[1]);
//		//	printf("data: %02x\r\n", data_decode[2]);
//		//	printf("data: %02x\r\n", data_decode[3]);
//		//
//
//		ota_packet_zero_fill(&rx_packet);
//		memcpy(&rx_packet, data_decode, peek_length);
//		uint16_t packet_size = MIN(rx_packet.size, ARRAY_LEN(rx_packet.data)-1);
//
		int packet_good = 0;
//		int j;
//
//		// search around a bit till the checksum matches up.  This is our "bit sync"
//		for(j = -5000; j < 5000; j+=50)
//		{
//			uint16_t decode_remainig_size = MAX(0, packet_size);
//			unsigned remaining_length = ota_length_encoded(decode_remainig_size);
//			uint8_t data_rx_remaining[remaining_length];
//			shannon_pop_data_demodulate(data, size, prnCodeStart+combDenseLength+j, data_rx_remaining, remaining_length, (scorePrn<0?1:0));
//			uint8_t data_decode_remaining[decode_remainig_size];
//			uint32_t data_decode_size_remaining;
//			decode_ota_bytes(data_rx_remaining, remaining_length, data_decode_remaining, &data_decode_size_remaining);
//
//			// the null terminated character is not transmitted
//			ota_packet_zero_fill_data(&rx_packet);
//			memcpy(((uint8_t*)&rx_packet), data_decode_remaining, decode_remainig_size);
//
//			if(ota_packet_checksum_good(&rx_packet))
//			{
//				packet_good = 1;
//				break;
//			}
//		}

		if( !packet_good )
		{
//			printf("Bad packet checksum\r\n");


			printf("Transmitting code only (testing ldpc rn)\r\n");
			char buf[128];
			snprintf(buf, 128, "{}");
			rpc->packet_tx(buf, strlen(buf), txTime, pitTxTime);




//			printf("Packet (still) says: ");
//
//			for(int k = 0; k < 40; k++ )
//			{
//				char c = rx_packet.data[k];
//				if( isprint(c) )
//				{
//					cout << c;
//				}
//				else
//				{
//					cout << '0';
//				}
//			}
//
//			cout << endl;

//			return;
		}
		else
		{

			if(rx_packet.data[ARRAY_LEN(rx_packet.data)-1] != '\0' )
			{
				printf("Packet c-string is not null terminated\r\n");
				return;
			}

			printf("Packet (offset %d) says: %s\r\n", j, rx_packet.data);

			//		printf("Packet (offset %d) says: ", j);
			//
			//		for(int k = 0; k < 40; k++ )
			//		{
			//			char c = rx_packet.data[k];
			//			if( isprint(c) )
			//			{
			//				cout << c;
			//			}
			//			else
			//			{
			//				cout << '0';
			//			}
			//		}
			//
			//		cout << endl;



			cout << "tpm: " << artemis_tpm << " pit: " << artemis_pit << " pps: " << artemis_pps << endl;

			if( rpc )
			{
				process_ota_packet(&rx_packet, txTime, pitTxTime, pitPrnCodeStart);
			}
			else
			{
				printf("Rpc pointer not set, skipping json parse\r\n");
			}
			printf("\r\n");

		} // packet good


		// regardless if checksum was good or bad, we've got a comb here.  now time to do fabric stuff (which is blocking?) after transmitting reply which is time sensative


		uint32_t rx_frac_int = (prnCodeStart - artemis_pps);

		ostringstream os;

		os << "{\"method\":\"packet_rx\",\"params\":[" << "\"" << pop_get_hostname() << "\"" << "," << artimes_pps_full_sec << "," << rx_frac_int << "]}";

		rpc->fabric->send("noc", os.str());


		// if we've found a comb, but there aren't enough samples ahead of us to represent the entire message

		uint32_t sampend=0;
		uint32_t end_comb_dense_length = comb_end[ARRAY_LEN(comb_end)-1];

		// find the edge right before the comb
		for(i = 0; i < size; i++)
		{
			if( data[i] < (end_comb_dense_length + prnEndCodeStartUnscaled) )
			{
				sampend = i;
			}
		}

		previous_run_offset = size - sampend;


	} // comb detected
	else
	{
		// nothing detected, throw away all of those samples
		previous_run_offset = size - final_sample;
	}


}

void PopPacketHandler::set_artimes_timers(uint32_t a_tpm, uint64_t a_pit, uint32_t a_pps)
{
	 boost::mutex::scoped_lock lock(timer_mtx);

	 uint32_t artemis_tpm;
	 uint64_t artemis_pit;
	 uint32_t artemis_pps;
	 uint64_t artimes_pps_full_sec;

	 artemis_tpm = a_tpm;
	 artemis_pit = a_pit;
	 artemis_pps = a_pps;

	 new_timers++;
	 artemis_tpm_start = -1;



	 // "now" in system time
	 PopTimestamp now = get_microsec_system_time();

	 // artemis_tpm is the most recent timer value, sent to us over uart asap.  we assume this is pretty accurately "now"

	 // this is how far we were into the second when we took a time reading.
	 double frac_since_pps = ((double)(artemis_tpm - artemis_pps))/ARTEMIS_CLOCK_SPEED_HZ;

	 // make timestamp
	 PopTimestamp delta_since_pps(frac_since_pps);

	 // offset as if we had taken system time reading at the edge of pps
	 now -= delta_since_pps;

	 // round
	 artimes_pps_full_sec = round(now.get_real_secs());


	 artemis_timers.push_back(boost::make_tuple(artemis_tpm, artemis_pit, artemis_pps, artimes_pps_full_sec));



//	 cout << "time delta: " << artemis_tpm - artemis_pps << endl;

}


} //namespace



#ifdef ZOMG_FALSE

		double fdev = 0.0008;
		double f;

		for( f = 1-fdev; f <= 1+fdev; f += fdev/100)
		{
			for(i = 0; i < alc; i++)
			{
				comb2[i] = round(comb[i] * f);
			}

			uint32_t prnCodeStartScale;
			int32_t scorePrnScale;
			uint32_t finalSampleShort = 0;

			prnCodeStartScale = pop_correlate_spool(data, size, comb2, alc, &scorePrnScale, &finalSampleShort);

			cout << "prnCodeScale: " << prnCodeStartScale << endl;
			cout << "ScoreScale: " << scorePrnScale << endl;

			if( abs(scorePrnScale) > abs(scorePrn) )
			{
				cout << "Better " << endl;
			}
			else
			{
				cout << "Worse " << endl;
			}

			cout << "delta: " << abs(scorePrnScale) - abs(scorePrn) << " " << f <<  endl;
		}




		while(1) {}

#endif
