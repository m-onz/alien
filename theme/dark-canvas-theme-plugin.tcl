namespace eval dark_canvas_theme {

    variable color_bg #202020
    variable color_fg #01c6e8
    variable color_hl_bg #400000
    variable color_hl_fg #01c6e8
    variable color_insert white
    variable color_sel #7760ff
    variable color_debug #00ff00

    variable color_log_fatal_fg #ffe0e8
    variable color_log_fatal_bg #d00
    variable color_log_err #d00
    variable color_log_normal $color_fg
    variable color_log_normal_sel_fg black
    variable color_log_debug #888888
    variable color_log_verbose #686868
    variable color_log_sel_bg $color_fg

    variable theme_name [option get . * TkTheme]

    proc is_black {color} {
        if {[lsearch {black #000 #000000} $color] >= 0} {
            return 1
        } else {
            return 0
        }
    }

    proc is_blue {color} {
        if {[lsearch {blue #00f #0000ff} $color] >= 0} {
            return 1
        } else {
            return 0
        }
    }

    proc is_offwhite {color} {
        if {$color eq "#fcfcfc"} {
            return 1
        } else {
            return 0
        }
    }

    proc is_white {color} {
        # Catch white and common light colors used in panels
        if {[lsearch {white #fff #ffffff #f0f0f0 #e0e0e0 #d0d0d0 #c0c0c0 #cccccc #eeeeee #fafafa} [string tolower $color]] >= 0} {
            return 1
        }
        # Also catch any very light color (RGB values all > 0xc0)
        if {[regexp {^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$} $color -> r g b]} {
            set r [expr 0x$r]
            set g [expr 0x$g]
            set b [expr 0x$b]
            if {$r > 192 && $g > 192 && $b > 192} {
                return 1
            }
        }
        return 0
    }

    proc is_bright {color} {
        # Catch any bright color that would look too light on dark background
        # Uses luminance calculation: L = 0.299*R + 0.587*G + 0.114*B
        set color_lower [string tolower $color]

        # Named bright colors
        if {[lsearch {white lime green yellow cyan magenta red orange pink lightgreen lightblue lightyellow} $color_lower] >= 0} {
            return 1
        }

        # Check hex colors by luminance
        if {[regexp {^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$} $color -> r g b]} {
            set r [expr 0x$r]
            set g [expr 0x$g]
            set b [expr 0x$b]
            # Calculate perceived luminance (0-255 scale)
            set lum [expr {0.299*$r + 0.587*$g + 0.114*$b}]
            # If luminance > 140, it's too bright for dark theme
            if {$lum > 140} {
                return 1
            }
        }
        # 3-char hex
        if {[regexp {^#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$} $color -> r g b]} {
            set r [expr 0x$r$r]
            set g [expr 0x$g$g]
            set b [expr 0x$b$b]
            set lum [expr {0.299*$r + 0.587*$g + 0.114*$b}]
            if {$lum > 140} {
                return 1
            }
        }
        return 0
    }

    proc color_canvas_item {canv item_type tags} {
        variable color_bg
        variable color_fg

        set tag [lindex $tags 0]

        catch {
            $canv itemconfigure $tag -outline $color_fg
        }

        if {[regexp {X[12]$} $tag]} {
            $canv itemconfigure $tag -fill $color_bg
        } elseif {[lsearch {line text} $item_type] >= 0} {
            $canv itemconfigure $tag -fill $color_fg
        } elseif {[lsearch $tags "x"] >= 0} {
        } elseif {
            [lsearch $tags "inlet"] >= 0 || [lsearch $tags "outlet"] >= 0
        } {
            $canv itemconfigure $tag -fill $color_fg
        } elseif {[lsearch $tags "array"] >= 0} {
            $canv itemconfigure $tag -fill $color_fg
        } elseif {[regexp {BASE\d*$} $tag]} {
            $canv itemconfigure $tag -fill $color_bg
        } elseif {[regexp {BUT$} $tag] && $item_type eq "oval"} {
            $canv itemconfigure $tag -fill $color_bg
        } elseif {[regexp {BUT0$} $tag] && $item_type eq "rectangle"} {
            $canv itemconfigure $tag -fill $color_fg -outline $color_fg
        } elseif {[regexp {BUT\d+$} $tag] && $item_type eq "rectangle"} {
            $canv itemconfigure $tag -fill $color_bg -outline $color_bg
        } elseif {$item_type eq "rectangle"} {
            # Handle panel/cnv objects - check if they have bright fill
            set fill_clr ""
            catch {set fill_clr [$canv itemcget $tag -fill]}
            if {[is_white $fill_clr] || [is_bright $fill_clr]} {
                $canv itemconfigure $tag -fill $color_bg -outline $color_bg
            }
        }
    }

    proc canvas_trace {cmd code result op} {
        variable color_bg
        variable color_fg
        variable color_sel

        if {$code != 0} {
            return
        }

        set canv [lindex $cmd 0]
        set canv_cmd [lindex $cmd 1]

        if {$canv_cmd eq "create"} {

            set tags_idx [lsearch $cmd -tags]

            if {$tags_idx >= 0} {
                incr tags_idx
                set tags [lindex $cmd $tags_idx]
                set tag [lindex $tags 0]
                set item_type [lindex $cmd 2]

                color_canvas_item $canv $item_type $tags
            }
        } elseif {$canv_cmd eq "itemconfigure"} {
            set tag [lindex $cmd 2]
            set fill_clr [$canv itemcget $tag -fill]
            set outline_clr ""

            catch {
                set outline_clr [$canv itemcget $tag -outline]
            }
            
            if {[is_black $fill_clr]} {
                $canv itemconfigure $tag -fill $color_fg
            } elseif {[is_blue $fill_clr]} {
                $canv itemconfigure $tag -fill $color_sel
            } elseif {[is_offwhite $fill_clr] || [is_white $fill_clr] || [is_bright $fill_clr]} {
                $canv itemconfigure $tag -fill $color_bg
            }

            if {[is_black $outline_clr]} {
                $canv itemconfigure $tag -outline $color_fg
            } elseif {[is_blue $outline_clr]} {
                $canv itemconfigure $tag -outline $color_sel
            } elseif {[is_offwhite $outline_clr] || [is_white $outline_clr] || [is_bright $outline_clr]} {
                $canv itemconfigure $tag -outline $color_bg
            }
        } 
    }

    proc canvas_created {cmd code result op} {
        variable color_bg
        variable color_hl_fg
        variable color_hl_bg
        variable color_insert

        if {$code != 0} {
            return
        }

        set container [lindex $cmd 1]
        set canv [tkcanvas_name $container]
        
        $canv configure -background $color_bg
        $canv configure -selectforeground $color_hl_fg
        $canv configure -selectbackground $color_hl_bg
        $canv configure -insertbackground $color_insert

        trace add execution $canv leave [namespace code canvas_trace]
    }

    trace add execution ::pdtk_canvas_new leave [namespace code canvas_created]

    ::.pdwindow.text configure -background $color_bg
    ::.pdwindow.text configure -selectbackground $color_log_sel_bg
    ::.pdwindow.text.internal tag configure log0\
        -foreground $color_log_fatal_fg\
        -background $color_log_fatal_bg
    ::.pdwindow.text.internal tag configure log1\
        -foreground $color_log_err
    ::.pdwindow.text.internal tag configure log2\
        -foreground $color_log_normal\
        -selectforeground $color_log_normal_sel_fg
    ::.pdwindow.text.internal tag configure log3\
        -foreground $color_log_debug

    for {set i 4} {$i <= 24} {incr i} {
        ::.pdwindow.text.internal tag configure log$i\
            -foreground $color_log_verbose
    }
}


