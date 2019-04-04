Purpose of this project is to reproduce the WeFDE information leakage results with code which is more user-friendly and expandable.

Directories:
 * info_leakage -- contains source code for re-creation of WeFDE analysis
 * preprocess -- revisions of author's original preprocessing scripts
 
 
Files:
 * info_leak.py -- main file of info_leakage project code
 * fingerprint_modeler.py -- class responsible for processing data for information leakage
 * mi_analyzer -- mutual information analyzer (for redundant feature identification and combined leakage measurements)
 
 
Order-of-Goals:
 1) Reproduce individual feature information leakage analysis
   * -> TODO: importance sampling for monte-carlo integration
 2) Reproduce feature clustering for dimension reduction
 3) Reproduce combined information leakage results 