# alien-theme-overlay-plugin.tcl
#
# The alien-theme canvas recolouring, but with an always-on TRANSPARENT canvas
# background so a GEM window behind the patch shows through your cables and
# boxes. Cyan strokes/text, dark boxes - identical to alien-theme, except the
# empty canvas is see-through and the patch window floats on top. For audio-
# visual performance: your GEM visuals behind, your live patch cables on top.
#
# Pure Tcl GUI plugin - no binaries, one package for every OS/CPU/float size.
# True per-pixel transparency is macOS-only (see README for other platforms).
#
# Install: copy this folder onto a Pd search path, e.g.
#   ~/Documents/Pd/externals/alien-theme-overlay-plugin/
# then RESTART Pd. Use only ONE of alien-theme-overlay / alien-theme at a time
# (both recolour the canvas).

namespace eval ::alien_theme_overlay {

    variable color_bg #202020
    variable color_fg #01c6e8
    variable color_hl_bg #400000
    variable color_hl_fg #01c6e8
    variable color_insert white
    variable color_sel #7760ff

    proc post {msg} {
        if {[catch {::pdwindow::post "alien-theme-overlay: $msg\n"}]} {
            catch {puts stderr "alien-theme-overlay: $msg"}
        }
    }

    proc is_black {color} {
        return [expr {[lsearch {black #000 #000000} $color] >= 0}]
    }
    proc is_blue {color} {
        return [expr {[lsearch {blue #00f #0000ff} $color] >= 0}]
    }
    proc is_offwhite {color} {
        return [expr {$color eq "#fcfcfc"}]
    }
    proc is_white {color} {
        if {[lsearch {white #fff #ffffff #f0f0f0 #e0e0e0 #d0d0d0 #c0c0c0 #cccccc #eeeeee #fafafa} [string tolower $color]] >= 0} {
            return 1
        }
        if {[regexp {^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$} $color -> r g b]} {
            if {[expr 0x$r] > 192 && [expr 0x$g] > 192 && [expr 0x$b] > 192} { return 1 }
        }
        return 0
    }
    proc is_bright {color} {
        set c [string tolower $color]
        if {[lsearch {white lime green yellow cyan magenta red orange pink lightgreen lightblue lightyellow} $c] >= 0} {
            return 1
        }
        if {[regexp {^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$} $c -> r g b]} {
            if {[expr {0.299*[expr 0x$r] + 0.587*[expr 0x$g] + 0.114*[expr 0x$b]}] > 140} { return 1 }
        }
        if {[regexp {^#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$} $c -> r g b]} {
            if {[expr {0.299*[expr 0x$r$r] + 0.587*[expr 0x$g$g] + 0.114*[expr 0x$b$b]}] > 140} { return 1 }
        }
        return 0
    }

    proc color_canvas_item {canv item_type tags} {
        variable color_bg
        variable color_fg
        set tag [lindex $tags 0]
        catch { $canv itemconfigure $tag -outline $color_fg }
        if {[regexp {X[12]$} $tag]} {
            $canv itemconfigure $tag -fill $color_bg
        } elseif {[lsearch {line text} $item_type] >= 0} {
            $canv itemconfigure $tag -fill $color_fg
        } elseif {[lsearch $tags "x"] >= 0} {
        } elseif {[lsearch $tags "inlet"] >= 0 || [lsearch $tags "outlet"] >= 0} {
            $canv itemconfigure $tag -fill $color_fg
        } elseif {[lsearch $tags "array"] >= 0} {
            $canv itemconfigure $tag -fill $color_fg
        } elseif {[regexp {BASE\d*$} $tag]} {
            $canv itemconfigure $tag -fill $color_bg
        } elseif {[regexp {BUT$} $tag] && $item_type eq "oval"} {
            $canv itemconfigure $tag -fill $color_bg
        } elseif {[regexp {BUT0$} $tag] && $item_type eq "rectangle"} {
            $canv itemconfigure $tag -fill $color_fg
            catch {$canv itemconfigure $tag -outline $color_fg}
        } elseif {[regexp {BUT\d+$} $tag] && $item_type eq "rectangle"} {
            $canv itemconfigure $tag -fill $color_bg
            catch {$canv itemconfigure $tag -outline $color_bg}
        } elseif {$item_type eq "rectangle"} {
            set fill_clr ""
            catch {set fill_clr [$canv itemcget $tag -fill]}
            if {[is_white $fill_clr] || [is_bright $fill_clr]} {
                $canv itemconfigure $tag -fill $color_bg
                catch {$canv itemconfigure $tag -outline $color_bg}
            }
        }
    }

    proc canvas_trace {cmd code result op} {
        variable color_bg
        variable color_fg
        variable color_sel
        if {$code != 0} { return }
        set canv [lindex $cmd 0]
        set canv_cmd [lindex $cmd 1]
        if {$canv_cmd eq "create"} {
            set tags_idx [lsearch $cmd -tags]
            if {$tags_idx >= 0} {
                incr tags_idx
                set tags [lindex $cmd $tags_idx]
                set item_type [lindex $cmd 2]
                color_canvas_item $canv $item_type $tags
            }
        } elseif {$canv_cmd eq "itemconfigure"} {
            set tag [lindex $cmd 2]
            set fill_clr [$canv itemcget $tag -fill]
            set outline_clr ""
            catch { set outline_clr [$canv itemcget $tag -outline] }
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

    # transparent canvas background so the GEM window behind shows through
    proc make_transparent {top canv} {
        variable color_bg
        if {$::windowingsystem eq "aqua"} {
            catch {wm attributes $top -transparent 1}
            catch {$top configure -background systemTransparent}
            catch {$canv configure -background systemTransparent}
        } else {
            # Tk has no per-pixel transparency off aqua; keep the dark theme bg
            catch {$canv configure -background $color_bg}
        }
        catch {wm attributes $top -topmost 1}
    }

    proc canvas_created {cmd code result op} {
        variable color_hl_fg
        variable color_hl_bg
        variable color_insert
        if {$code != 0} { return }
        set container [lindex $cmd 1]
        set canv [tkcanvas_name $container]
        $canv configure -selectforeground $color_hl_fg
        $canv configure -selectbackground $color_hl_bg
        $canv configure -insertbackground $color_insert
        make_transparent $container $canv
        trace add execution $canv leave [namespace code canvas_trace]
    }

    # theme the Pd main window too (matches alien-theme)
    catch {
        ::.pdwindow.text configure -background $color_bg
        ::.pdwindow.text configure -selectbackground $color_fg
        ::.pdwindow.text.internal tag configure log0 -foreground #ffe0e8 -background #d00
        ::.pdwindow.text.internal tag configure log1 -foreground #d00
        ::.pdwindow.text.internal tag configure log2 -foreground $color_fg -selectforeground black
        ::.pdwindow.text.internal tag configure log3 -foreground #888888
        for {set i 4} {$i <= 24} {incr i} {
            ::.pdwindow.text.internal tag configure log$i -foreground #686868
        }
    }

    trace add execution ::pdtk_canvas_new leave [namespace code canvas_created]

    post "loaded - transparent patch canvas over GEM (cyan theme)"
}
