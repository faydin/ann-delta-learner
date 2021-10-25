import tkinter as tk # for GUI
import numpy as np   # for matrix multiplication


# Parameters
gui_height = 400
gui_width = 600
error_limit = 0.001
counter_limit = 100000
LearningRate = 10
gui_point_size = 3
visualize_density = 8 #lower~more dense

number_of_outputs=3
number_of_layer=1



root = tk.Tk()
inputs = np.empty((2,0), int)
classes = np.empty((number_of_outputs,0), int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# GUI Elements
label = tk.Label(root)
label.pack()
label.config(text = 'Merhaba! Başlamak için üstteki menüden sınıf seçin...')
canvas = tk.Canvas(root, height = gui_height, width = gui_width, bg = 'white')
canvas.create_line(0, gui_height/2, gui_width, gui_height/2, dash = (5,2))  # x-axis
canvas.create_line(gui_width/2, 0, gui_width/2, gui_height, dash = (5,2))  # y-axis
canvas.pack()

# Functions
def class_pick(color):
    global fill_color
    fill_color = color
    label.config(text = 'Örnek noktaları koordinat düzlemi üzerine tıklayarak seçiniz',fg = color)

def set_parameters():
    label.config(text = 'Bu özellik henüz geliştirilmedi. Parametleri kod üzerinden ayarlayınız.', fg = 'black')

def button(event):
    global inputs, classes
    x, y = event.x, event.y
    try:
        canvas.create_oval(x-gui_point_size, y-gui_point_size, x+gui_point_size, y+gui_point_size, width = 0, fill = fill_color)
        x_coor = (x-int(gui_width/2))/10
        y_coor = -(y-int(gui_height/2))/10
        inputs=np.append(inputs, [[x_coor],[y_coor]],axis=1)
        classes=np.append(classes,([[0],[0],[0]] if fill_color == 'red' else
                                   [[0],[0],[1]] if fill_color == 'blue' else
                                   [[0],[1],[0]] if fill_color == 'orange' else
                                   [[0],[1],[1]] if fill_color == 'green' else 
                                   [[1],[0],[0]] if fill_color == 'purple' else
                                   0),axis=1)
    except:
        label.config(text = 'Lütfen sınıf seçtikten sonra noktaları oluşturunuz!',fg = 'indigo')
canvas.bind('<Button-1>', button)

def classify():
    try:
        global w
        inputs_norm = inputs/np.linalg.norm(inputs)
        w = np.random.rand(number_of_outputs,len(inputs_norm)+1)
        #w = np.zeros((number_of_outputs,len(inputs_norm)+1))
        error = 1
        counter = 0

        while error>error_limit and counter<counter_limit:
            error = 0
            counter+= 1
            for x in range(inputs_norm.shape[1]):
                net = np.matmul(w,np.vstack([1,inputs_norm[:,[x]]]))
                try:
                    output = sigmoid(net)
                except OverflowError: # exp(709>) = hafıza hatası
                    output = 0
                for k in range(number_of_outputs):
                    w[k,:] += LearningRate*(classes[k,x]-output[k])*np.hstack([1,inputs_norm[:,x]])
                error += 1/2*np.sum((classes[:,[x]]-output)**2)
            print(error) 
        label.config(text = "Sınıflandırma {} döngü kullanılarak tamamlandı!".format(counter),fg = 'green')
    except IndexError:
        label.config(text = 'Henüz örnek noktalar seçilmemiş!', fg = 'indigo')
def visualize():
    label.config(text = 'Görselliştirme işlemine başlandı...')
    print(w)
    try:
        w
        output1=0
        output2=0
        output3=0
        output4=0

        for x in range(int(gui_width/visualize_density)):
            for y in range(int(gui_height/visualize_density)):

                m = x*visualize_density
                n = y*visualize_density
                x_coor = ((m-int(gui_width/2))/10)/np.linalg.norm(inputs)
                y_coor = -(n-int(gui_height/2))/10/np.linalg.norm(inputs)

                net1 = np.matmul(w[0,:],[[1],[x_coor],[y_coor]])
                try:
                    output1 = sigmoid(net1)
                except OverflowError:
                    output1=0               
                net2 = np.matmul(w[1,:],[[1],[x_coor],[y_coor]])
                try:
                    output2 = sigmoid(net2)
                except OverflowError:
                    output2=0               
                net3 = np.matmul(w[2,:],[[1],[x_coor],[y_coor]])
                try:
                    output3 = sigmoid(net3)
                except OverflowError:
                    output3=0
                
           
                if output1<0.5 and output2<0.5 and output3<0.5:     
                    canvas.create_oval(m-2, n-2, m+2, n+2, width = 1, outline  = 'red')
                elif output1<0.5 and output2<0.5 and output3>0.5:     
                    canvas.create_oval(m-2, n-2, m+2, n+2, width = 1, outline  = 'blue')
                elif output1<0.5 and output2>0.5 and output3<0.5:     
                    canvas.create_oval(m-2, n-2, m+2, n+2, width = 1, outline  = 'orange')
                elif output1<0.5 and output2>0.5 and output3>0.5:     
                    canvas.create_oval(m-2, n-2, m+2, n+2, width = 1, outline  = 'green')
                elif output1>0.5:     
                    canvas.create_oval(m-2, n-2, m+2, n+2, width = 1, outline  = 'purple')




        label.config(text = 'Görselleştirme tamamlandı!', fg = 'green')
    except NameError:
        label.config(text = 'Henüz sınıflandırma yapılmamış!', fg = 'indigo')

# Menubar
menubar = tk.Menu(root)

fileMenu = tk.Menu(menubar)
fileMenu.add_command(label = "Kırmızı", command = lambda: class_pick('red'))
fileMenu.add_command(label = "Mavi", command = lambda: class_pick('blue'))
fileMenu.add_command(label = "Turuncu", command = lambda: class_pick('orange'))
fileMenu.add_command(label = "Yeşil", command = lambda: class_pick('green'))
fileMenu.add_command(label = "Mor", command = lambda: class_pick('purple'))
fileMenu.add_command(label = "Sarı", command = lambda: class_pick('yellow'))
fileMenu.add_command(label = "Siyah", command = lambda: class_pick('black'))
fileMenu.add_command(label = "Çıkış", command = root.quit)
menubar.add_cascade(label = "Sınıf Seçimi", menu = fileMenu)

functionsMenu = tk.Menu(menubar)
functionsMenu.add_command(label = "Sınıfları Ayır", command = classify)
functionsMenu.add_command(label = "Sınıfları Görselleştir", command = visualize)
menubar.add_cascade(label = "İşlemler", menu = functionsMenu)

settingsMenu = tk.Menu(menubar)
settingsMenu.add_command(label = "Parametreleri Ayarla", command = set_parameters)
menubar.add_cascade(label = "Ayarlar", menu = settingsMenu)

root.config(menu = menubar)

root.mainloop()

