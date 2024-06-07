# import matplotlib.pyplot as plt

# # Data for plotting
# epochs = list(range(1, 41))
# train_loss = [
#     0.02757733197027522, 0.018157631081554704, 0.015844959113573986, 0.01419147799245553,
#     0.013023648802318082, 0.01220382428439064, 0.01159404504361208, 0.011120826333425801,
#     0.010837869579998846, 0.010589826343623782, 0.010419303029100784, 0.010335825529298964,
#     0.010232801253359184, 0.010181657592603522, 0.010115353316610795, 0.010076153905590518,
#     0.010041874920438643, 0.010012098513643226, 0.01000533404158556, 0.009988624115111665,
#     0.009966648289218102, 0.009962525527505732, 0.009954453811223398, 0.009947102157496373,
#     0.009942275630498989, 0.009935078635432646, 0.0099248577272882, 0.009922882525901358,
#     0.009921297273404696, 0.009910494243887338, 0.009898546051407943, 0.00989296251557848,
#     0.009897198649560898, 0.009906110151756277, 0.009903403994021235, 0.0098871340645665,
#     0.00989793295989915, 0.009899286782958889, 0.009886145794999544, 0.009889249768337997
# ]
# train_avg_relative_error = [
#     45.41027889741403, 31.89536511104132, 26.373109113918904, 21.7056061186038,
#     18.60038818347888, 15.732785515595873, 18.157774163579507, 11.067364805747996,
#     13.200167772930467, 19.133175955040738, 9.407207336446156, 10.983231740129948,
#     8.706101090382525, 9.039172525138554, 8.222826084745321, 8.94019133995467,
#     9.692761374975104, 8.768711517574475, 8.381289606451269, 8.532636800617379,
#     8.818691507951403, 8.411376502732738, 7.688335992305641, 8.130350935118388,
#     7.643323937854554, 8.584042981755704, 7.111319362904689, 6.8127275320853755,
#     7.181564093412453, 7.867687502166598, 7.165921012010177, 6.555873959537316,
#     7.1026960733942595, 6.093678771032928, 7.240985040895117, 6.601766868713978,
#     6.318661895723218, 7.273888171537646, 7.250016415547742, 6.43846789129848
# ]
# test_loss = [
#     0.01653147121915738, 0.014533413545105433, 0.013199315038963369, 0.012332921703567165,
#     0.011633452543291869, 0.011143354274116151, 0.010803958438197502, 0.010620042677414862,
#     0.010383916550901724, 0.010295868262131901, 0.01014392516843421, 0.010138647354824957,
#     0.01001284550178689, 0.00997796790614201, 0.009946417406891625, 0.009918288134806942,
#     0.009912174259526738, 0.009887023860715728, 0.00986485739709159, 0.009854840401855879,
#     0.009820698868087428, 0.009875881184124222, 0.00983426763137261, 0.009832854420301074,
#     0.009826667609299348, 0.009845201734444224, 0.00985732295277856, 0.009830867971242498,
#     0.009831008949593758, 0.009830993695017153, 0.009854969032255592, 0.009856528694103722,
#     0.00988240802664315, 0.009855522642986942, 0.009848419711073645, 0.009857744711664533,
#     0.009852759810272099, 0.009853166118398059, 0.009835951837151892, 0.009840645934825958
# ]
# test_avg_relative_error = [
#     22.204694142773263, 17.92539904825477, 19.361831495583143, 14.08328480115762,
#     12.059601086201054, 12.225584220445027, 9.988566801928506, 8.633505902984187,
#     8.155859831690732, 6.92135384368312, 6.855247558638601, 6.614182925080983,
#     7.127461620922192, 9.028108154614541, 6.931631293880215, 7.997534940316957,
#     8.620263967520783, 8.566773741228971, 7.673910145248685, 8.724250396570469,
#     6.912309638498977, 9.322123511044948, 7.885466516277859, 7.684243565303208,
#     7.7863469790985205, 8.69425636683079, 8.731083948198549, 7.538918545705875,
#     8.146402137268831, 8.147485077971279, 8.244822941316517, 8.084169335884622,
#     8.958989646863623, 8.345736190262183, 8.623158195273374, 8.735491743210295,
#     8.581942468755148, 7.99970048937186, 7.5713149299041556, 8.130224873018063
# ]

# # Plotting Train Loss and Test Loss
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_loss, label='Train Loss', color='b')
# plt.plot(epochs, test_loss, label='Test Loss', color='r')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Train Loss and Test Loss Over Epochs')
# plt.legend()

# # Plotting Train Avg Relative Error and Test Avg Relative Error
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_avg_relative_error, label='Train Avg Relative Error', color='b')
# plt.plot(epochs, test_avg_relative_error, label='Test Avg Relative Error', color='r')
# plt.xlabel('Epochs')
# plt.ylabel('Avg Relative Error')
# plt.title('Train and Test Avg Relative Error Over Epochs')
# plt.legend()

# plt.tight_layout()
# plt.show()

import pandas as pd

# Data for 15 epochs
data_15_epochs = {
    "Epoch": list(range(1, 16)),
    "Train Loss": [
        0.040427954966564705, 0.02594273198577663, 0.02585451651887338, 0.02556432125727602,
        0.025157761153607235, 0.024752665520685523, 0.024411996152343338, 0.024161528314358072,
        0.024013535098873474, 0.02380141987705989, 0.02377929648355763, 0.023783515792401667,
        0.023731992763817242, 0.02377619822172228, 0.023785158185340376
    ],
    "Train Avg Relative Error": [
        25.995110005603774, 14.040112648664723, 14.610055443937487, 13.31367250518837,
        11.555175240887847, 10.435450313587573, 10.174420256335074, 8.516763276100702,
        7.855486912638234, 8.059955403984723, 8.322289878129109, 8.296435475536514,
        8.076312982214553, 7.638721356668798, 7.835292070461996
    ],
    "Test Loss": [
        0.1184492913543635, 0.1421965839519486, 0.061242617286979335, 0.03841760457686418,
        0.029326601888441346, 0.024941840932485998, 0.02483540534302762, 0.02481537434783741,
        0.024219288400550435, 0.02432558434359848, 0.024047109254918025, 0.023498649484051524,
        0.02348891816447767, 0.023728940178744138, 0.023967071452256555
    ],
    "Test Avg Relative Error": [
        68.49181788388941, 98.34590755511354, 50.20060985137158, 29.282611222433477,
        21.323712296488964, 10.280356783968584, 10.709910895686848, 7.661303967856096,
        6.531000838826324, 9.497906292290105, 9.546157794533073, 7.141451431159176,
        7.841373884726356, 7.642663002651916, 6.352648038209033
    ]
}

# Create a DataFrame
df_15_epochs = pd.DataFrame(data_15_epochs)

import matplotlib.pyplot as plt

# Plot Train Loss and Test Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df_15_epochs["Epoch"], df_15_epochs["Train Loss"], label="Train Loss", color="blue")
plt.plot(df_15_epochs["Epoch"], df_15_epochs["Test Loss"], label="Test Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss and Test Loss Over Epochs")
plt.legend()

# Plot Train Avg Relative Error and Test Avg Relative Error
plt.subplot(1, 2, 2)
plt.plot(df_15_epochs["Epoch"], df_15_epochs["Train Avg Relative Error"], label="Train Avg Relative Error", color="blue")
plt.plot(df_15_epochs["Epoch"], df_15_epochs["Test Avg Relative Error"], label="Test Avg Relative Error", color="red")
plt.xlabel("Epochs")
plt.ylabel("Avg Relative Error")
plt.title("Train and Test Avg Relative Error Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()