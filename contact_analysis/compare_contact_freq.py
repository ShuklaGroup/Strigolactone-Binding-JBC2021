"""Compare contact probabilities and identify most residues with highest difference"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=20)

def get_sequence_alignment(nres1, shift=4):
    #For now this is temporarily hacked purely for AtD14 (seq1)/ShHTL7(seq2)
    #Returns len seq1 vector containing corresponding resids from seq2

    alignment = []
    for i in range(nres1):
        if i+shift < 169:
            alignment.append(i-2+shift)
        else:
            alignment.append(i-1+shift)

    return alignment

def compute_change(vec1, vec2, alignment, scale=True):

    if scale:
        vec1 = normalize_probs(vec1)
        vec2 = normalize_probs(vec2)
   
    change = np.zeros(np.size(vec1) - 3)

    print(np.shape(vec1))
    print(np.shape(vec2))

    for i in range(len(change)):
        print(i)
        print(alignment[i])
        change[i] = vec1[i] - vec2[alignment[i]]

    return change

def id_mutations(changes, seq1, seq2, alignment, n=10):
    highest = np.argsort(changes)[-n::]
    lowest = np.argsort(changes)[0:n]
    
    print("More ligand contact in D14")
    mutations_highest = []
    for i in highest:
        print(changes[i])
        mutation = "%s%s%s"%(seq1[i], i+5, seq2[alignment[i]])
        mutations_highest.append(mutation)
        print(mutation)

    print("Less ligand contact in D14")
    mutations_lowest = []
    for i in lowest:
        print(changes[i])
        mutation = "%s%s%s"%(seq1[i], i+5, seq2[alignment[i]])
        mutations_lowest.append(mutation)
        print(mutation)

    return mutations_highest, mutations_lowest

def scale_vector(vec):
    #Change everything to 0-to-1 scale
    return vec/np.max(vec)

def normalize_probs(vec):
    return vec/np.sum(vec)

def plot_contact_freq_comparison(vec1, vec2, alignment, pocket, entrance, other):

    align_array = np.array(alignment, dtype=int)
    vec1_pocket = vec1[pocket]
    vec2_pocket = vec2[list(align_array[pocket])]
    vec1_entrance = vec1[entrance]
    vec2_entrance = vec2[list(align_array[entrance])]
    vec1_other = vec1[other]
    vec2_other = vec2[list(align_array[other])]

    #vec2_use = vec2[alignment]
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(np.linspace(0,1), np.linspace(0,1), color='k', linestyle=":")
    plt.scatter(vec1_pocket, vec2_pocket, s=12, color='r', marker='o', label="Pocket")
    plt.scatter(vec1_entrance, vec2_entrance, s=12, color='b', marker='^', label="Entrance")
    plt.scatter(vec1_other, vec2_other, s=12, color='g', marker='s', label="Other")
    plt.xlabel("AtD14 Contact Probability")
    plt.ylabel("ShHTL7 Contact Probability")
    plt.xlim(0,1)
    plt.ylim(0,1)
    leg = plt.legend(fancybox=True, frameon=True, edgecolor='k', markerfirst=False, fontsize=16)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_facecolor('none')
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    plt.savefig("contact_comparison_colored.svg", transparent=True)

if __name__=="__main__":
    
    #AtD14_dir = "/home/jiming/Storage/AtD14_binding_analysis/contact_prob_analysis"
    #ShHTL7_dir = "/home/jiming/Documents/Striga/ShHTL7_binding/contact_prob_analysis"

    AtD14_lig_contact = np.load("%s/eq_contact_probs_final.npy"%AtD14_dir)
    ShHTL7_lig_contact = np.load("%s/eq_contact_probs_final.npy"%ShHTL7_dir)

    nres_AtD14 = np.shape(AtD14_lig_contact1)[0]
    nres_ShHTL7 = np.shape(ShHTL7_lig_contact1)[0]

    contacts_to_plot_D14 = np.zeros((nres_AtD14*3, 2))
    contacts_to_plot_HTL7 = np.zeros((nres_ShHTL7*3, 2))

    np.save("contact_list_D14.npy", contacts_to_plot_D14)
    np.save("contact_list_HTL7.npy", contacts_to_plot_HTL7)

    probs_AtD14 = np.zeros(nres_AtD14*3)
    probs_ShHTL7 = np.zeros(nres_ShHTL7*3)

    np.save("probs_D14.npy", probs_AtD14)
    np.save("probs_HTL7.npy", probs_ShHTL7)

    align = get_sequence_alignment(np.size(AtD14_lig_contact))
    print(align[92])
    print(align[213])
    print(align[242])

    change = compute_change(AtD14_lig_contact, ShHTL7_lig_contact, align, scale=False)

    np.save("change.npy",change_all)
    print(np.max(change))
    print(np.min(change))
    np.save("weighted_contact_prob_change_normed_4Acutoff.npy", change)

    AtD14_seq = "NILEALNVRVVGTGDRILFLAHGFGTDQSAWHLILPYFTQNYRVVLYDLVCAGSVNPDYFDFNRYTTLDPYVDDLLNIVDSLGIQNCAYVGHSVSAMIGIIASIRRPELFSKLILIGFSPRFLNDEDYHGGFEEGEIEKVFSAMEANYEAWVHGFAPLAVGADVPAAVREFSRTLFNMRPDISLFVSRTVFNSDLRGVLGLVRVPTCVIQTAKDVSVPASVAEYLRSHLGGDTTVETLKTEGHLPQLSAPAQLAQFLRRALP"
    ShHTL7_seq = "MSSIGLAHNVTILGSGETTVVLGHGYGTDQSVWKLLVPYLVDDYKVLLYDHMGAGTTNPDYFDFDRYSSLEGYSYDLIAILEEFQVSKCIYVGHSMSSMAAAVASIFRPDLFHKLVMISPTPRLINTEEYYGGFEQKVMDETLRSLDENFKSLSLGTAPLLLACDLESAAMQEYCRTLFNMRPDIACCITRMICGLDLRPYLGHVTVPCHIIQSSNDIMVPVAVGEYLRKNLGGPSVVEVMPTEGHLPHLSMPEVTIPVVLRHIRQDIT"
    id_mutations(change, AtD14_seq, ShHTL7_seq, align, n=20)

