#include <iostream>
#include <sstream>
#include "TTree.h"
#include "TH1.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLorentzVector.h"

#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include <TChain.h>
#include "TMath.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TLatex.h"
#include "THStack.h"

using namespace std;

void makeBrazilPlot(TString channel="", TString method="mu", TString ext=""){
    gStyle->SetOptStat(0);
    gROOT->SetBatch();
    const int points=2;
    int mass[points]	={900, 1200};
    double T_xs[points] 	={90.3, 11.8}; //cross section
    /*
       float Tprime600_xs= 0.1764;
       float Tprime625_xs= 0.1489;
       float Tprime650_xs= 0.1213;
       float Tprime675_xs= 0.1050;
       float Tprime700_xs= 0.0886;
       float Tprime800_xs= 0.0459;
       float Tprime900_xs= 0.0251;
       float Tprime1000_xs= 0.0145;
       float Tprime1100_xs= 0.00867;
       float Tprime1200_xs= 0.00536;
       */
    string mass_str[points]={"900", "1200"};
    TString file_path="";
    float lumi=137;
    double x[points];
    double y[points];
    double y_Stat[points];
    double eyl[points];
    double eyh[points];
    double ey2l[points];
    double ey2h[points];

    double exl_[points];
    double exh_[points];
    double ex2l_[points];
    double ex2h_[points];

    double eyl_[points];
    double eyh_[points];
    double ey2l_[points];
    double ey2h_[points];

    double obsl[points];

    cout<<"Limit with Systematics\n";
    for(int i=0; i<points; i++)
    {
        double xs_X_Br_=1;
        if(method=="xs_X_Br")  xs_X_Br_ = T_xs[i]*1000*0.00223; //in fb
        else xs_X_Br_ =1;
        TFile *file_ = new TFile(file_path + "higgsCombine" +  mass_str[i] + ".AsymptoticLimits.mH120.root", "READ");
        TTree* tree_ = (TTree*)file_->Get("limit");
        Int_t nentries=(Int_t)tree_->GetEntries();
        //cout<<"nentries=========="<<nentries<<endl;
        Double_t qlimit;
        tree_->SetBranchAddress("limit", &qlimit);
        tree_->SetBranchStatus("*", 1);
        x[i] = mass[i];
        for(int ievent=0; ievent <points ; ievent++)
        {
            tree_-> GetEntry( ievent );
            //                x[ievent] = mass[i];
            if (ievent==0) ey2l[i] = qlimit*xs_X_Br_;// cout<<"ey2l[ievent] "<<qlimit<<endl;
            else if (ievent==1) eyl[i] = qlimit*xs_X_Br_;
            else if (ievent==2) {y[i]= qlimit*xs_X_Br_; cout<<"Tprime"<<mass[i]<<" mean	=	"<<qlimit<<endl;}
            else if (ievent==3) eyh[i] = qlimit*xs_X_Br_;
            else if (ievent==4) ey2h[i] = qlimit*xs_X_Br_;
            //else if (ievent==5) {obsl[i] = qlimit*xs_X_Br_;}
            //		else	cout<<"Limit ---  "<<qlimit<<endl;
        }
        //cout<<"y[i]   "<<y[i]<<endl;

    }
    //====================================================================================================
    for(int j=0; j<points; j++)
    {
        //cout<<"y[j]   "<<y[j]<<endl;
        //cout<<"ey2l[j] "<<ey2l[j]<<endl;
        ex2l_[j]=0.;
        exl_[j]=0.;
        exh_[j]=0.;
        ex2h_[j]=0.;
        ey2l_[j] = y[j]-ey2l[j];
        eyl_[j] = y[j]-eyl[j];
        eyh_[j] = eyh[j]-y[j];
        ey2h_[j] = ey2h[j]-y[j];

        y[j] *= T_xs[j];
        ey2l_[j] *= T_xs[j];
        eyl_[j] *= T_xs[j];
        eyh_[j] *= T_xs[j];
        ey2h_[j] *= T_xs[j];
    }
    //====================================================================================================
    for(int j=0; j<points; j++)
    {
        y_Stat[j] *= T_xs[j];
    }

    TGraph* gStat = new TGraph(points, x, y_Stat);
    gStat->SetLineColor(kRed);
    TGraphAsymmErrors* ge = new TGraphAsymmErrors(points, x, y,exl_,exh_,eyl_,eyh_);
    ge->SetFillColor(kGreen);
    TGraphAsymmErrors* ge2 = new TGraphAsymmErrors(points, x, y, ex2l_,ex2h_,ey2l_,ey2h_);
    ge2->SetFillColor(kYellow);
    ge2->GetXaxis()->SetTitle("T' mass(GeV)");
    ge2->GetYaxis()->SetRangeUser(0.002,50);
    //ge2->GetYaxis()->SetTitle("95\% CL limit on #mu");
    if(method=="xs_X_Br") ge2->GetYaxis()->SetTitle("95\% CL limit on T' in fb");
    else ge2->GetYaxis()->SetTitle("95 \% CL limit on  #sigma(pp #rightarrow T' #bar{T}')");
    ge2->SetTitle("");
    TGraph* g = new TGraph(points, x, y);
    g->SetLineWidth(2);

    //TGraph* g_Stat = new TGraph(points, x, y_Stat);
    //g_Stat->SetLineWidth(2);
    //g_Stat->SetLineStyle(2);

    TGraph* g_obs = new TGraph(points, x, T_xs);
    g_obs->SetLineWidth(2);
    g_obs->SetLineStyle(2);

    TLatex * tex1 = new TLatex(0.10, 0.95,TString("#bf{CMS} #it{Premilinary}"));
    tex1->SetNDC();
    tex1->SetTextAlign(13);
    tex1->SetTextFont(42);
    tex1->SetTextSize(0.03);
    tex1->SetLineWidth(2);

    TLatex * tex2 = new TLatex(0.76, 0.95,TString(Form("%.0f fb^{-1} (14 TeV)",lumi)));
    tex2->SetNDC();
    tex2->SetTextAlign(13);
    tex2->SetTextFont(42);
    tex2->SetTextSize(0.03);
    tex2->SetLineWidth(2);

    TLegend *leg1 = new TLegend(0.3, 0.75, 0.88, 0.88, "");
    leg1->SetNColumns(2);
    //leg1->SetFillColor(0);
    //leg1->SetFillStyle(0);
    leg1->SetBorderSize(0);
    leg1->SetTextSize(0.03);

    leg1->AddEntry(g,  "Expected (Cross-section)"		,"l");
    leg1->AddEntry(g_obs,  "Theoretical (Cross-section)"        ,"l");
    leg1->AddEntry(gStat,  "Stat. Only (Cross-section)"        ,"l");
    //leg1->AddEntry(graph, "Expected(H #rightarrow bb)", "l");
    leg1->AddEntry(ge, "#pm1 std. deviation" ,"f");
    leg1->AddEntry(ge2,"#pm2 std. deviation" ,"f");


    //TLegend *leg2 = new TLegend(0.1, 0.7, 0.35, 0.9, "");
    //leg1->AddEntry(graph_mva1,  "Expected #mu in MVA[600,700]"        ,"l");
    //leg1->AddEntry(graph_mva2,  "Expected #mu in MVA[800,1000]"        ,"l");
    //leg1->AddEntry(graph_mva3,  "Expected #mu in MVA[1100,1200]"        ,"l");


    TCanvas* c41 = new TCanvas("c41","c41",200,10,600,400);
    c41->cd();
    c41->SetGridx();
    c41->SetGridy();
    c41->SetLogy();
    ge2->Draw("a3");
    ge->Draw("Same3l");
    g->Draw("SAME");
    g_obs->Draw("SAME");
    gStat->Draw("SAME");
    //graph_mva1->Draw("SAME");
    //graph_mva2->Draw("SAME");
    //graph_mva3->Draw("SAME");
    c41->Update();
    //c41->SetGridx();
    //c41->SetGridy();
    tex1->Draw();
    tex2->Draw();
    leg1->Draw();
    //leg2->Draw();

    c41->SaveAs("/eos/user/s/sraj/"+ method + channel + ext + "200percent.pdf");
    c41->SaveAs("/eos/user/s/sraj/"+ method + channel + ext + "200percent.png");
    //c41->SaveAs("/eos/user/k/kpal/www/i4DAPNVShN/"+ method + channel + ext + "200percent.root");
}
