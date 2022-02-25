# HiggsAnalysis
**Step by Step Higgs Analysis:**

*Step 1*:-
  ## Store the Output from the testing File after training as a root File.
 
 As it is done in the training and testing of signal and backround as in the file, [Storing of ROOT File](https://github.com/raj2022/HiggsAnalysis/tree/main/Codes)
 
 Codes to Store file as the ROOT file after testing on the training sample:
 
      from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile,TH1D
      from ROOT import gROOT, gBenchmark
      f = TFile("/eos/home-s/sraj/M.Sc._Thesis/Plot_M.Sc._thesis/ROOT_output/MyrootFile_after_training_Tprime_600_all_five_background_test_with_TPrime1200.root", "RECREATE")
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
      
*step 4*:-
 ## Use Comined limit tools to Run the Datacards
   It will give output as a .root file of the corresponding datacars.
   text2workspace.py "Datacard.txt" (both should be in the same folder or specify the path properly)
      e.g. "text2workspace.py Datacard_signal_tprime_1200_background_all_five_backgrounds_except_thq.txt" 
    further, 
    **use the combine tools on the "Output of Datacard as .root file" -M AsymptoticLimits** to obtain the CLs.
    e.g.
    combine Datacard_signal_tprime_900_background_all_five_backgrounds_except_thq.root -M AsymptoticLimits
 
 This will give us a output as "higgsCombine1200.AsymptoticLimits.mH120.root" which will be used in making the [Brazil_plot](https://github.com/raj2022/HiggsAnalysis/tree/main/Brazil_plots)
 *Step 5*:-
 
