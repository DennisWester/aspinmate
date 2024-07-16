#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Dennis Westerbeck"
# __credits__ = 
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Dennis Westerbeck"
__email__ = "dwesterbeck@physik.uni-bielefeld.de"
__status__ = "Production"

title  = "Molmag - simulator"

import customtkinter as ctk
import tkinter as tk
import numpy as np
import time
import sys
import os
from spin_class import *
from spin_animation import interactive_video


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

nf = 24
cf = 32
fname = "Arial"

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    


class NumberEntryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        def on_closing():
            if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.quit()
                sys.exit()

        self.protocol("WM_DELETE_WINDOW", on_closing)
        self.title(title)
        self.geometry("1200x800")
        
        self.validate_cmd = self.register(self.validate_integer)

        self.frame1 = ctk.CTkFrame(self)
        self.frame1.pack(pady=20)
         # Adding the heading above the entries
        self.heading = ctk.CTkLabel(self.frame1, text="Spins (doubled: 1-> s=1/2)\n fill up with 0 if you want < 4 spins", font=(fname, cf))
        self.heading.grid(row=0, column=0, columnspan=4, pady=(0, 10))

        self.entries = []
        for i in range(4):
            entry = ctk.CTkEntry(self.frame1, width=50, validate="key", validatecommand=(self.validate_cmd, '%P'))
            entry.grid(row=1, column=i, padx=10, pady= 10)
            self.entries.append(entry)
        self.checkbox = ctk.CTkSwitch(self.frame1, text="include dipolar?", command=self.set_dipolar, font=(fname, nf), variable=tk.BooleanVar(), onvalue=True, offvalue=False)
        self.checkbox.grid(row=2, column=0, columnspan=4 )


        # Adding the button to go to the next window
        self.frame1_1 = ctk.CTkFrame(self)
        self.frame1_1.place(anchor="se", relx=1, rely=1, x=-20, y=-20)
        self.next_button = ctk.CTkButton(self.frame1_1, text="Go to the interaction menu", command=self.go_to_second_window)
        self.next_button.grid(row=0, column=0, pady=20, sticky="e") 
        self.sys_choice_id = -1
        self.changing_value_id = -1
        self.slider_value = 0
        self.initials = []
        self.dipolar = False
        self.gfactor = 2.0

    def validate_integer(self, value_if_allowed):
        if value_if_allowed == "" or value_if_allowed.isdigit():
            return True
        else:
            return False
        
    def set_dipolar(self):
        self.dipolar = self.checkbox.get()
        
    def validate_float(self, value_if_allowed):
        if value_if_allowed == "" or value_if_allowed == "-":
            return True
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False
        
    def get_entries(self):
        self.spins = []
        for entry in self.entries:
            value = entry.get()
            if value.isdigit():
                if int(value) > 0:
                    self.spins.append(int(value))
            else:
                tk.messagebox.showerror("Invalid input", "Please enter valid integers in all fields.")
                return False
        return True
        



    ########################################################################################################
    '''SECOND WINDOW'''
    ########################################################################################################
    def go_to_second_window(self):
        if self.get_entries():
            self.frame1.pack_forget()
            self.frame1_1.pack_forget()
            self.frame1_1.destroy()
            
            self.frame2 = ctk.CTkFrame(self)
            self.frame2.pack(pady=20)
            
            self.label = ctk.CTkLabel(self.frame2, text="Your Spins", font=(fname, cf))
            self.label.pack(pady=10)
            spinstring = "Spins: "
            for i in range(len(self.spins)):
                if (self.spins[i] % 2) == 0:
                    spinstring += str(self.spins[i]//2) + "  "
                else:
                    spinstring += str(self.spins[i]) + "/2  "
            self.label_spins = ctk.CTkLabel(self.frame2, text=spinstring, font=(fname, nf))
            self.label_spins.pack(pady=20)
            
        
            validate_cmd = self.register(self.validate_float)
            
            '''Frame 2_2 is for the system choice and the dipolar distance'''
            self.frame2_2 = ctk.CTkFrame(self)
            self.frame2_2.pack(pady=20)
            

            # Add a button to frame2 that brings you back to frame1
            # self.frame2back = ctk.CTkFrame(self)
            # self.frame2back.grid(row=3, column=0, pady=20, padx=20, sticky="se")
            # self.back_button = ctk.CTkButton(self.frame2back, text="Back", command=self.go_to_first_window)
            # self.back_button.grid(row=0, column=0)


            #Dipolar Distance
            if self.dipolar:
                label_dipolar = ctk.CTkLabel(self.frame2_2, text="Set the distance (A)", font=(fname, nf))
                label_dipolar.pack(padx=20, pady=10)
                entry_dip = ctk.CTkEntry(self.frame2_2, width=70, validate="key", validatecommand=(validate_cmd, '%P'))
                entry_dip.pack(padx=20, pady=10)
                self.dist_entry = entry_dip

            if len(self.spins) == 4:
                system_dropdown = ctk.CTkOptionMenu(self.frame2_2,
                                       values=["Ring", "Chain", "Tetrahedron", "Butterfly"],
                                       command=self.system_choice)
                system_dropdown.pack(padx=20, pady=10)
                system_dropdown.set("Choose a system")

            if len(self.spins) == 3:
                system_dropdown = ctk.CTkOptionMenu(self.frame2_2,
                                       values=["Ring", "Chain"],
                                       command=self.system_choice)    
                system_dropdown.pack(padx=20, pady=10)
                system_dropdown.set("Choose a system")

            if len(self.spins) == 2:
                system_dropdown_label = ctk.CTkLabel(self.frame2_2, text="Dimer", font=(fname, nf))
                system_dropdown_label.pack(padx=20, pady=10)
                self.sys_choice_id = 1

            #Add , "ZFS direction \u03C6" to the values of the dropdown if phi should be included
            var_change_dropdown = ctk.CTkOptionMenu(self.frame2_2,
                                       values=["Heisenberg interaction J", "ZFS strength D", "ZFS direction \u03D1"],
                                       command=self.changing_value)
            var_change_dropdown.pack(padx=20, pady=10)
            var_change_dropdown.set("Choose the variable to change")
            
            #Adding the range of the changing interaction parameters
            self.frame2_3 = ctk.CTkFrame(self)
            self.frame2_3.pack(pady=20)

            #Adding the the parameters (start, stop, num_steps) for the changing value
            self.lin_paras = []
            lin_paranames = ["range", "num_steps"]
            lin_desc = ctk.CTkLabel(self.frame2_3, text="Select the change for the frames", font=(fname, cf))
            lin_desc.grid(row=3, column=0, columnspan=2, pady=10)
            for i in range(2):
                entry_lin = ctk.CTkEntry(self.frame2_3, width=70, validate="key", validatecommand=(validate_cmd, '%P'))
                entry_lin.grid(row=5, column=i, pady=10)
                label_lin = ctk.CTkLabel(self.frame2_3, text=lin_paranames[i], font=(fname, nf))
                label_lin.grid(row=4, column=i, pady=10)
                self.lin_paras.append(entry_lin)

            #Warning if spins equal 1 that the zfs has no effect
            # Identify indices of spins with value 1 and format them for display
            mask = [i for i, spin in enumerate(self.spins) if spin == 1]
            if mask:
                indices_str = ", ".join(str(i + 1) for i in mask)  # Adding 1 to make indices human-readable (1-based)
                warning_text = f"Warning: ZFS has no effect for spin 1/2 (i = {indices_str})"
                self.warning_label = ctk.CTkLabel(self.frame2_3, text=warning_text, font=(fname, nf))
                self.warning_label.grid(row=6, column=0, columnspan=2, pady=10)
            
            #Adding a button that brings you to the next window
            self.frame_to_3rd = ctk.CTkFrame(self)
            self.frame_to_3rd.place(anchor="se", relx=1, rely=1, x=-20, y=-20)
            self.next_button2 = ctk.CTkButton(self.frame_to_3rd, text="Set initial values", command=self.go_to_third_window)
            self.next_button2.grid(row=0, column=0, pady=20, sticky="e")

    def changing_value(self, choice):
        if choice == "Heisenberg interaction J":
            self.changing_value_id = 0
        elif choice == "ZFS strength D":
            self.changing_value_id = 1
        elif choice == "ZFS direction \u03D1":
            self.changing_value_id = 2
        # elif choice == "ZFS direction \u03C6":
        #     self.changing_value_id = 3

    def init_spin_system(self):
        if self.sys_choice_id == 0:
            self.spin_system = SpinRing(self.spins, self.gfactor, self.distance, dipolar=self.dipolar)
        elif self.sys_choice_id == 1:
            self.spin_system = SpinChain(self.spins, self.gfactor, self.distance, dipolar=self.dipolar)
        elif self.sys_choice_id == 2:
            self.spin_system = SpinTetrahedron(self.spins, self.gfactor, self.distance, dipolar=self.dipolar)
        elif self.sys_choice_id == 3:
            self.spin_system = SpinButterfly(self.spins, self.gfactor, (self.distance, self.distance*1.3), dipolar=self.dipolar)

    def system_choice(self, choice):
        if choice == "Ring":
            self.sys_choice_id = 0
        elif choice == "Chain":
            self.sys_choice_id = 1
        elif choice == "Tetrahedron":
            self.sys_choice_id = 2
        elif choice == "Butterfly":
            self.sys_choice_id = 3
        else:
            self.sys_choice_id = -1


    def expected_timefunc(self, dim, steps):
        x = dim * (steps/50)
        exp = 110* np.exp(0.0082*x)
        if x < 30:#For very low dimensions and steps the exponential fit is bad
            exp = 30
        return exp
    
    def calc_expected_time(self):
        expected_timetable = [60, 600, 1800, 3600, 43200, 86400]
        num_steps = self.lin_paras[1]
        hilbertdim = np.prod([self.spins[i]+1 for i in range(len(self.spins))])
        time = self.expected_timefunc(num_steps, hilbertdim)
        ret = 0
        for i in range(len(expected_timetable)):
            if time < expected_timetable[i]:
                ret = i
                break
        if time > expected_timetable[-1]:
           ret = -1
        return ret
    
    def restart_application(self):
        # Close the current window
        self.destroy()
        os.execl(sys.executable, sys.executable, *sys.argv)  # Restart the application


    def check_system(self):
        if self.sys_choice_id == -1:
            tk.messagebox.showerror("Invalid input", "Please choose a system.")
            return False
        if self.changing_value_id == -1:
            tk.messagebox.showerror("Invalid input", "Please choose a variable to change.")
            return False
        #check the dipolar interaction
        if self.dipolar:
            self.distance = self.dist_entry.get()
            if self.distance == "":
                self.dipolar = False
                self.distance = 100
            else:
                self.distance = float(self.distance)
            if self.distance < 0:
                tk.messagebox.showerror("Invalid input", "Please fill in a positive number for the distance.")
                return False
        if self.dipolar == False:
            self.distance = 100
        #check whethter self.lin_paras is filled with numbers
        for idx, entry in enumerate(self.lin_paras):
            value = entry.get()
            if value == "":
                tk.messagebox.showerror("Invalid input", "Please fill numbers in all fields.")
                return False 
            if is_float(value):
                self.lin_paras[idx] = float(value)
        self.lin_paras[1] = int(round(self.lin_paras[1]))
        #Check the estimated time of the calculation
        exp_time = self.calc_expected_time()
        exptime_names = ["less than 1 minute", "few minutes", "less than half an hour", "less than an hour", "couple of hours", "about a day"]
        if exp_time == -1:
            response = tk.messagebox.askquestion("", "Expected calculation time: " + exptime_names[exp_time] + " Are you sure to continue?") 
            if response == 'no':
                # Call a method to restart the application or show the first window again
                self.restart_application()   
                return False       
        else:
            tk.messagebox.showinfo("", "Expected calculation time: " + exptime_names[exp_time])
            
        return True
    



    ########################################################################################################
    '''THIRD WINDOW'''
    ########################################################################################################
    def go_to_third_window(self):
        if self.check_system():
            self.frame2.pack_forget()
            self.frame2.destroy()
            self.frame2_2.pack_forget()
            self.frame2_2.destroy()
            self.frame2_3.pack_forget()
            self.frame2_3.destroy()
            self.frame_to_3rd.pack_forget()
            self.frame_to_3rd.destroy()

            self.init_spin_system() #Initialize the spin system
            # Register the validation function
            self.heis_entries = []
            self.zfs_entries = []
            zfsint_label = ["D", "\u03D1", "\u03C6"]
            heisint_label = [str(i) for i in self.spin_system.heis_int]
            validate_cmd = self.register(self.validate_float)

            self.frame3 = ctk.CTkFrame(self)
            self.frame3.pack(pady=20)
            #Set the maximum column length for the labels in order to look nice
            if self.changing_value_id == 0:
                max_col_length = max(len(heisint_label) * 2, len(zfsint_label))
            else:
                max_col_length = max(len(heisint_label), len(zfsint_label) * 2)
            self.label = ctk.CTkLabel(self.frame3, text="Please select the initial parameters", font=(fname, cf))
            self.label.grid(row=0, column=0, columnspan=max_col_length, pady=10)
            self.label = ctk.CTkLabel(self.frame3, text="Only checked paras will be changed", font=(fname, nf))
            self.label.grid(row=1, column=0, columnspan=max_col_length, pady=10)
            self.label = ctk.CTkLabel(self.frame3, text="----------------------------------", font=(fname, nf))
            self.label.grid(row=2, column=0, columnspan=max_col_length, pady=10)
            
            #################### Heisenberg initial values ####################
            self.mask = []
            self.checkbuttons = []
            self.heis_label = ctk.CTkLabel(self.frame3, text="Heisenberg interaction J", font=(fname, nf))
            self.heis_label.grid(row=3, column=0, columnspan=max_col_length - (self.changing_value == 0), pady=10)
            
            if self.changing_value_id == 0:
                self.master_checkbox_var = tk.IntVar()
                self.master_checkbox = ctk.CTkCheckBox(self.frame3, text="Select All", variable=self.master_checkbox_var, command=lambda: self.update_all_checkboxes(self.master_checkbox_var.get()))
                self.master_checkbox.grid(row=3, column=max_col_length-1, pady=10, sticky='e')

            hbidx = 0
            for i in range(len(self.spin_system.heis_int)):
                if self.changing_value_id == 0:
                    hbidx = i*2
                    bvar = tk.BooleanVar()
                    heis_check = ctk.CTkCheckBox(self.frame3, text="", variable=bvar, onvalue=True, offvalue=False)
                    heis_check.grid(row=5, column=hbidx+1, sticky="w")
                    self.mask.append(bvar)
                    self.checkbuttons.append(heis_check)
                else:
                    cspan = 1
                    hbidx = i
                label_heis = ctk.CTkLabel(self.frame3, text=heisint_label[i], font=(fname, nf))
                label_heis.grid(row=4, column=hbidx, pady=10)
                entry_heis = ctk.CTkEntry(self.frame3, width=50, validate="key", validatecommand=(validate_cmd, '%P'))
                entry_heis.grid(row=5, column=hbidx, pady=10)
                
                self.heis_entries.append(entry_heis)

            #################### ZFS initial values ####################
            self.zfs_label = ctk.CTkLabel(self.frame3, text="zero field splitting (ZFS)", font=(fname, nf))
            self.zfs_label.grid(row=6, column=0, columnspan=max_col_length-1, pady=10)
            if self.changing_value_id != 0:
                self.master_checkbox_var = tk.IntVar()
                self.master_checkbox = ctk.CTkCheckBox(self.frame3, text="Select All", variable=self.master_checkbox_var, command=lambda: self.update_all_checkboxes(self.master_checkbox_var.get()))
                self.master_checkbox.grid(row=6, column=max_col_length-1, pady=10, sticky='w')
            if self.changing_value_id == 0:
                indent_entry = 100
            else:
                indent_entry = self.changing_value_id
            for i in range(3):
                label_zfs = ctk.CTkLabel(self.frame3, text=zfsint_label[i], font=(fname, nf))
                label_zfs.grid(row=7, column=i + 1, pady=10)
                for j in range(self.spin_system.spinnum):
                    label_spin = ctk.CTkLabel(self.frame3, text=str(j+1), font=(fname, nf-2))
                    label_spin.grid(row=8+j, column=0, pady=10)
                    if self.changing_value_id != 0 and i == 0:
                        bvar = tk.BooleanVar()
                        zfs_check = ctk.CTkCheckBox(self.frame3, text="", variable=bvar, onvalue=True, offvalue=False)
                        zfs_check.grid(row=8+j, column=indent_entry+1, sticky="w")
                        self.mask.append(bvar)
                        self.checkbuttons.append(zfs_check)
                    if self.spins[j] == 1:
                        entry_zfs = ctk.CTkEntry(self.frame3, width=50, validate="key", validatecommand=(validate_cmd, '%P'), placeholder_text="0")
                    else:
                        entry_zfs = ctk.CTkEntry(self.frame3, width=50, validate="key", validatecommand=(validate_cmd, '%P'))
                    entry_zfs.grid(row=8+j, column=i + 1, pady=10)
                    self.zfs_entries.append(entry_zfs)

            #################### Filename chooser ####################
            self.create_filename()
            self.create_name("")
            self.frame_file = ctk.CTkFrame(self)
            self.frame_file.pack(pady=20)
            self.file_label = ctk.CTkLabel(self.frame_file, text="The gif will be named: " + self.desc_filename, font=(fname, nf-2))
            self.file_label.grid(row=0, column=0, columnspan=2,  pady=10)


            # self.frame_file = ctk.CTkFrame(self)
            # self.frame_file.pack(pady=20)
            # self.file_label = ctk.CTkLabel(self.frame_file, text="Select a name for the gif.", font=(fname, nf))
            # self.file_label.grid(row=0, column=0, columnspan=2,  pady=10)
            # self.file_label = ctk.CTkLabel(self.frame_file, text="Enter nothing to select placeholder", font=(fname, nf-2))
            # self.file_label.grid(row=1, column=0, columnspan=2, pady=10)
            # self.entry_filename = ctk.CTkEntry(self.frame_file, width=300, placeholder_text=self.desc_filename)
            # self.entry_filename.grid(row=2, column=0, pady=10)
            # self.create_name(self.entry_filename.get())
            # self.gif_ending = ctk.CTkLabel(self.frame_file, text=".gif", font=(fname, nf-2))
            # self.gif_ending.grid(row=2, column=1, pady=10, sticky="w")

            #################### Button to create video ####################
            self.frame_video = ctk.CTkFrame(self)
            self.frame_video.place(anchor="se", relx=1, rely=1, x=-20, y=-20)
            self.video_button = ctk.CTkButton(self.frame_video, text="Create Video", command=self.go_to_final_window)
            self.video_button.grid(row=0, column=0, pady=20, sticky="e")


    # Function to update all checkboxes based on the master checkbox
    def update_all_checkboxes(self, master_value):
        for var in self.mask:
            var.set(master_value)
        for cb in self.checkbuttons:
            cb.update()
    
    def create_filename(self):
        systemnames = ["Ring", "Chain", "Tetrahedron", "Butterfly"]
        changingvalues = ["J", "D", "theta", "phi"]
        dipstring = ""
        if self.dipolar:
            dipstring = "_dip"
        self.desc_filename = systemnames[self.sys_choice_id] + "_s" + ''.join(str(x) for x in self.spins) + "_change" + changingvalues[self.changing_value_id] + dipstring
    
    def create_name(self, name):    
        if name != "":
            self.desc_filename = name            
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
        # Define the relative path to the 'gif' folder
        gif_folder = os.path.join(script_dir, 'spin_gifs')
        
        # Create the 'gif' folder if it doesn't exist
        if not os.path.exists(gif_folder):
            os.makedirs(gif_folder)
        self.filename = os.path.join(gif_folder, self.desc_filename)


    def check_initials(self):
        self.heis_arr = []
        self.zfs_arr = []
        #Check whether at least one checkbox is checked
        if not any([var.get() for var in self.mask]):
            tk.messagebox.showerror("Invalid input", "Please select at least one parameter to change.")
            return False
        for i in range(len(self.mask)):
            self.mask[i] = self.mask[i].get()
        #Check whether all heisenberg entries are filled with numbers
        for entry in self.heis_entries:
            value = entry.get()
            if value == "":
                self.heis_arr.append(0.0)
            if is_float(value):
                self.heis_arr.append(float(value))
            else:
                tk.messagebox.showerror("Invalid input", "Please fill in valid J numbers.")
                return False
        #Check whether all zfs entries are filled with numbers
        for entry in self.zfs_entries:
            value = entry.get()   
            if value == "":
                self.zfs_arr.append(0.0)
            elif is_float(value):
                self.zfs_arr.append(float(value))
            else:
                tk.messagebox.showerror("Invalid input", "Please fill in valid ZFS numbers.")
                return False
        return True

    ########################################################################################################
    '''FINAL WINDOW - VIDEO CREATION'''
    ########################################################################################################
    def go_to_final_window(self):
        if self.check_initials():
            tk.messagebox.showinfo("", "The video will be created now. This may take a while.")
            self.frame3.pack_forget()
            self.frame3.destroy()
            self.frame_file.pack_forget()
            self.frame_file.destroy()
            self.frame_video.pack_forget()
            self.frame_video.destroy()
            self.create_name("")
            self.call_interact_vid()
            self.frame4 = ctk.CTkFrame(self)
            self.frame4.pack(pady=20)
            self.label = ctk.CTkLabel(self.frame4, text="Video created!", font=(fname, cf))
            self.label.pack(pady=10)

            self.frame_close = ctk.CTkFrame(self)
            self.frame_close.place(anchor="se", relx=1, rely=1, x=-20, y=-20)
            self.close_button = ctk.CTkButton(self.frame_close, text="Close", command=self.close_app)
            self.close_button.grid(row=0, column=0)


    def call_interact_vid(self):
        self.zfs_arr = np.insert(self.zfs_arr, self.spin_system.spinnum, [0]*self.spin_system.spinnum)
        zfs_arr = np.reshape(self.zfs_arr, (4, self.spin_system.spinnum))
        interactive_video(self.spin_system, self.lin_paras, self.changing_value_id, self.mask, zfs_arr, self.heis_arr, self.filename, self.sys_choice_id, self.distance)

    
    def close_app(self):
        self.quit()
        sys.exit()


if __name__ == "__main__":
    app = NumberEntryApp()
    app.mainloop()
            
