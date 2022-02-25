# HiggsAnalysis
**Step by Step Higgs Analysis:**

*Step 1*:-
  ## Store the Output from the testing File after training as a root File.
 
 As it is done in the training and testing of signal and backround as in the file, [Storing of ROOT File](https://github.com/raj2022/HiggsAnalysis/tree/main/Codes)
 
 Codes to Store file as the ROOT file after testing on the training sample:
 
      from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile,TH1D
      from ROOT import gROOT, gBenchmark
      f = TFile("/eos/home-    s/sraj/M.Sc._Thesis/Plot_M.Sc._thesis/ROOT_output/MyrootFile_after_training_Tprime_600_all_five_background_test_with_TPrime1200.root", "RECREATE")
       # tree = TTree("root",  )
       # How do we get that to create and fill with the background and signal
       h_bak = TH1D("background", "background", 100, 0,1)
       h_sig = TH1D("signal", "signal", 100, 0, 1)
       h_sum_all = TH1D("data_obs", "data_obs", 100, 0, 1)
       for i in tBkg:
           h_bak.Fill(i)
       for j in tSig:
           h_sig.Fill(j)
       h_sum_all.Add(h_bak) 
       h_sum_all.Add(h_sig)



       f.Write( )
       f.Close()

       
#tree = TTree("root",  )

*Step 2*:-
   ## Prepare the Datacard
 
 The data card should be prepare as just like the in [file](https://github.com/raj2022/HiggsAnalysis/blob/main/DataCards/datacard_practice_1.txt).
 Also a example for the signal and background seperation [as in the file](https://github.com/raj2022/HiggsAnalysis/blob/main/DataCards/Datacard_signal_tprime_900_background_ttgg.txt).

*step 3*:-
 ## Create the CMSSw env
     Inside Terminal(lxplus):
      1. cd CMSSW_10_2_13/src 
      2. cmsenv

